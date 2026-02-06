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


def extraer_texto_pdf(data: bytes) -> str:
    texto = ""
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            texto += page.extract_text() or ""
    return texto

def extraer_texto_docx(data: bytes) -> str:
    doc = Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)

def obtener_mantenimientos_jotform(limit=5):
    FORM_ID = os.getenv("JOTFORM_FORM_ID")
    API_KEY = os.getenv("JOTFORM_API_KEY")

    if not FORM_ID or not API_KEY:
        return ""

    url = f"https://api.jotform.com/form/{FORM_ID}/submissions?limit={limit}&orderby=created_at"
    headers = {
        "APIKEY": API_KEY
    }

    r = requests.get(url, headers=headers)
    data = r.json()

    if "content" not in data or not data["content"]:
        return ""

    texto = ""
    for i, sub in enumerate(data["content"], start=1):
        texto += f"\n--- MANTENIMIENTO #{i} ---\n"
        for _, ans in sub["answers"].items():
            pregunta = ans.get("text", "")
            respuesta = ans.get("answer", "")
            texto += f"{pregunta}: {respuesta}\n"

    return texto

# 1. Configura tu IA (Asegúrate de poner tu API Key real aquí)
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name='gemini-2.5-flash-lite',

generation_config={"temperature": 0.7},
system_instruction=
        """
Eres un asistente general en español.
Puedes responder preguntas, explicar conceptos y analizar documentos.

REGLAS:
- Responde siempre en ESPAÑOL.
- Si el usuario adjunta un documento, úsalo como CONTEXTO para responder.
- No resumas documentos automáticamente, a menos que el usuario lo solicite.
- Si la pregunta se refiere al contenido del documento, apóyate en él.
-si no hay documento, responde correctamente la pregunta que te realicen
"""
    )

app = FastAPI()

# Permisos para que no se bloquee el navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/jotform/test")
def test_jotform():
    try:
        url = "https://api.jotform.com/user"
        headers = {
            "APIKEY": os.getenv("JOTFORM_API_KEY")
        }

        r = requests.get(url, headers=headers)
        return r.json()

    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(texto: str = Form(...), archivo: Optional[UploadFile] = File(None)):
    try:
        texto_documento = ""
        contexto_jotform = ""

        # 1️⃣ Si hay archivo, se usa
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

            elif mime.startswith("image/"):
                contenido = [
                    texto,
                    {"mime_type": mime, "data": data}
                ]
                response = model.generate_content(contenido)
                return {"respuesta": response.text}

            else:
                raise HTTPException(status_code=400, detail="Tipo de archivo no soportado")

        # 2️⃣ Si NO hay archivo → consultar Jotform
        if not texto_documento.strip():
            contexto_jotform = obtener_mantenimientos_jotform()

        # 3️⃣ Construir prompt inteligente
        if texto_documento.strip():
            prompt = f"""
PREGUNTA DEL USUARIO:
{texto}

DOCUMENTO ADJUNTO:
{texto_documento}

Responde usando el documento.
"""
        elif contexto_jotform.strip():
            prompt = f"""
Eres un asistente de mantenimiento industrial.

REGISTROS REALES DE MANTENIMIENTO (JOTFORM):
{contexto_jotform}

Con base SOLO en esta información, responde la pregunta:
{texto}
"""
        else:
            prompt = texto

        response = model.generate_content(prompt)
        return {"respuesta": response.text}

    except Exception as e:
        print("ERROR:", e)
        return {"respuesta": "Ocurrió un error procesando la información."}
@app.get("/")
def home():
    return {"status": "Servidor de IA Activo"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
    