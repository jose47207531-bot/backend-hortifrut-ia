import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import pdfplumber
from docx import Document
from io import BytesIO
import uvicorn
import requests
import pandas as pd
import io
import pytesseract
from PIL import Image
import docx

# ===============================
# CONFIGURACIÓN GOOGLE SHEET
# ===============================

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1oEVKH1SxHDJtwSx9y3sy1Ui12CqvCWdRTb9bEe_w4D8/export?format=csv&gid=1960130423"

def obtener_mantenimientos_google_sheet(busqueda: str = "") -> str:
    try:
        response = requests.get(GOOGLE_SHEET_CSV_URL)
        response.raise_for_status()

        df_total = pd.read_csv(io.StringIO(response.text))

        if df_total.empty:
            return ""

        # 🔎 Filtrar por números si el usuario menciona alguno
        if any(char.isdigit() for char in busqueda):
            palabras = busqueda.split()
            for p in palabras:
                if p.isdigit():
                    filtro = df_total.astype(str).apply(
                        lambda x: x.str.contains(p, case=False)
                    ).any(axis=1)
                    df_total = df_total[filtro]

        return df_total.tail(15).to_string(index=False)

    except Exception as e:
        print("Error leyendo Google Sheets:", e)
        return ""

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
# FILTRO PARA USAR GOOGLE SHEET
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
        "orden",
        "orden de trabajo",
        "registro",
        "formulario"
    ]

    return any(p in texto for p in palabras_clave)

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
- Si la pregunta es sobre mantenimiento y hay datos del Sheet, úsalos.
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

@app.get("/")
def home():
    return {"status": "Servidor IA Activo"}

@app.post("/chat")
async def chat(
    texto: str = Form(...),
    archivo: Optional[UploadFile] = File(None)
):
    try:
        texto_documento = ""
        contexto_sheet = ""

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

        # 2️⃣ GOOGLE SHEET SOLO SI ES NECESARIO
        if not texto_documento.strip() and requiere_jotform(texto):
            contexto_sheet = obtener_mantenimientos_google_sheet(texto)

        # 3️⃣ CONSTRUCCIÓN DEL PROMPT
        if texto_documento.strip():
            prompt = f"""
PREGUNTA:
{texto}

DOCUMENTO:
{texto_documento}
"""
        elif contexto_sheet.strip():
            prompt = f"""
REGISTROS REALES DE MANTENIMIENTO:
{contexto_sheet}

Con base SOLO en esta información responde:
{texto}
"""
        else:
            prompt = texto

        # 4️⃣ RESPUESTA GEMINI
        response = model.generate_content(prompt)

        return {"respuesta": response.text}

    except Exception as e:
        print("ERROR:", e)
        return {"respuesta": "Ocurrió un error procesando la información."}

# ===============================
# MAIN
# ===============================

@app.post("/leer-archivo")
async def leer_archivo(file: UploadFile = File(...)):

    contenido = ""

    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                contenido += page.extract_text() + "\n"

    elif file.filename.endswith(".docx"):
        doc = docx.Document(file.file)
        for para in doc.paragraphs:
            contenido += para.text + "\n"

    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)
        contenido = df.to_string()

    elif file.filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file.file)
        contenido = pytesseract.image_to_string(image)

    else:
        return {"error": "Formato no soportado"}

    response = model.generate_content(f"Analiza este documento:\n{contenido}")

    return {"respuesta": response.text}

# ===============================
# HOME
# ===============================

@app.get("/")
def home():
    return {"status": "Servidor IA Empresarial + Motor Reglas + Gemini activo"}