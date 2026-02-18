import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pdfplumber
from docx import Document
from io import BytesIO
import uvicorn
import requests
from fastapi import Request


# ===============================
# UTILIDADES DOCUMENTOS
# ===============================

def extraer_texto_pdf(data: bytes) -> str:
    texto = ""
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            texto += page.extract_text() or ""
    return texto


def extraer_texto_docx(data: bytes) -> str:
    doc = Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


# ===============================
# FILTRO PARA USAR JOTFORM
# ===============================

def requiere_jotform(texto: str) -> bool:
    texto = texto.lower()

    palabras_clave = [
        "mantenimiento",
        "último mantenimiento",
        "ultimo mantenimiento",
        "repuesto",
        "repuestos",
        "falla",
        "fallas",
        "equipo",
        "máquina",
        "maquina",
        "inspección",
        "inspeccion",
        "orden de trabajo",
        "registro",
        "formulario"
    ]

    return any(p in texto for p in palabras_clave)


# ===============================
# JOTFORM
# ===============================

def obtener_forms_jotform():
    r = requests.get(
        "https://api.jotform.com/user/forms",
        headers={"APIKEY": os.getenv("JOTFORM_API_KEY")},
        timeout=10
    )
    data = r.json()
    return data.get("content", [])


def obtener_mantenimientos_todos_forms(max_forms=5, limit_por_form=3):
    forms = obtener_forms_jotform()[:max_forms]
    headers = {"APIKEY": os.getenv("JOTFORM_API_KEY")}

    contexto = ""

    for form in forms:
        form_id = form.get("id")
        form_title = form.get("title", "Formulario sin nombre")

        url = (
            f"https://api.jotform.com/form/{form_id}/submissions"
            f"?limit={limit_por_form}&orderby=created_at"
        )

        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()

        if "content" not in data or not data["content"]:
            continue

        contexto += f"\n===== FORMULARIO: {form_title} =====\n"

        for sub in data["content"]:
            for ans in sub.get("answers", {}).values():
                pregunta = ans.get("text", "")
                respuesta = ans.get("answer", "")
                contexto += f"{pregunta}: {respuesta}\n"

    return contexto.strip()


# ===============================
# IA GEMINI
# ===============================

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    generation_config={"temperature": 0.7},
    system_instruction="""
Eres un asistente general en español.

REGLAS:
- Responde siempre en ESPAÑOL.
- Si el usuario adjunta un documento, úsalo como CONTEXTO.
- Si la pregunta es sobre mantenimiento y hay datos, úsalos.
- Si no hay información suficiente, dilo claramente.
"""
)

# ===============================
# FASTAPI
# ===============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# ENDPOINTS
# ===============================
# ===============================
# ENDPOINT PARA WEBHOOK (PASO 1)
# ===============================

@app.post("/webhook/jotform")
async def recibir_webhook(request: Request):
    try:
        # 1. Recibir los datos del formulario
        form_data = await request.form()
        datos = dict(form_data)
        
        # 2. Imprimir en los logs de Render para verificar
        print(f"Nueva entrada recibida de: {datos.get('formTitle', 'Formulario Desconocido')}")
        
        # 3. (Opcional) Guardar en un archivo local para que Gemini lo lea luego
        # Nota: En Render, los archivos locales se borran al reiniciar. 
        # Esto es solo para probar la conexión.
        with open("consolidado.txt", "a") as f:
            f.write(f"\nORDEN: {datos.get('control_id', 'S/N')} | DATOS: {datos}\n")

        return {"status": "success"}
    except Exception as e:
        print(f"Error en Webhook: {e}")
        return {"status": "error", "message": str(e)}
    

@app.get("/")
def home():
    return {"status": "Servidor de IA Activo"}


@app.get("/jotform/ping")
def jotform_ping():
    r = requests.get(
        "https://api.jotform.com/user",
        headers={"APIKEY": os.getenv("JOTFORM_API_KEY")}
    )
    return r.json()


@app.post("/chat")
async def chat(
    texto: str = Form(...),
    archivo: Optional[UploadFile] = File(None)
):
    try:
        texto_documento = ""
        contexto_jotform = ""

        # 1️⃣ DOCUMENTOS
        if archivo:
            data = await archivo.read()
            mime = archivo.content_type

            if mime == "application/pdf":
                texto_documento = extraer_texto_pdf(data)

            elif mime in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword"
            ]:
                texto_documento = extraer_texto_docx(data)

            else:
                raise HTTPException(
                    status_code=400,
                    detail="Tipo de archivo no soportado"
                )

        # 2️⃣ JOTFORM SOLO SI ES NECESARIO
        if not texto_documento.strip() and requiere_jotform(texto):
            contexto_jotform = obtener_mantenimientos_todos_forms()

        # 3️⃣ PROMPT
        if texto_documento.strip():
            prompt = f"""
PREGUNTA:
{texto}

DOCUMENTO:
{texto_documento}
"""
        elif contexto_jotform.strip():
            prompt = f"""
REGISTROS REALES DE MANTENIMIENTO:
{contexto_jotform}

Con base SOLO en esta información responde:
{texto}
"""
        else:
            prompt = texto

        response = model.generate_content(prompt)
        return {"respuesta": response.text}

    except Exception as e:
        print("ERROR:", e)
        return {"respuesta": "Ocurrió un error procesando la información."}


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)))