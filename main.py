import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import pdfplumber
from docx import Document
from io import BytesIO
import requests
import pandas as pd
import io
import pytesseract
from PIL import Image
from collections import defaultdict

# ===============================
# CONFIGURACIÓN GEMINI
# ===============================

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    generation_config={"temperature": 0.4},
    system_instruction="""
Eres un asistente empresarial especializado en mantenimiento industrial.

Reglas:
- Responde siempre en español.
- Usa datos estructurados si existen.
- Si no hay datos suficientes, dilo claramente.
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
# MEMORIA CONVERSACIONAL
# ===============================

memoria_conversacion = defaultdict(list)

# ===============================
# GOOGLE SHEET
# ===============================

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1oEVKH1SxHDJtwSx9y3sy1Ui12CqvCWdRTb9bEe_w4D8/export?format=csv&gid=1960130423"

def buscar_texto_libre(df, consulta):
    palabras = consulta.lower().split()

    filtro = df.astype(str).apply(
        lambda fila: any(
            palabra in " ".join(fila).lower()
            for palabra in palabras
        ),
        axis=1
    )

    return df[filtro]

def obtener_mantenimientos_google_sheet(busqueda: str = "") -> str:
    try:
        response = requests.get(GOOGLE_SHEET_CSV_URL)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))

        if df.empty:
            return ""

        # 🔎 Búsqueda inteligente dentro de frases
        df_filtrado = buscar_texto_libre(df, busqueda)

        if df_filtrado.empty:
            return ""

        # 🔥 REDUCCIÓN DE TOKENS: solo columnas importantes
        columnas_clave = [
            col for col in df.columns
            if any(k in col.lower() for k in [
                "orden",
                "descripcion",
                "observacion",
                "responsable",
                "fecha",
                "trabajo"
            ])
        ]

        if columnas_clave:
            df_filtrado = df_filtrado[columnas_clave]

        return df_filtrado.head(5).to_string(index=False)

    except Exception as e:
        print("Error leyendo Google Sheets:", e)
        return ""

# ===============================
# DOCUMENTOS
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
# FILTRO PARA USAR SHEET
# ===============================

def requiere_sheet(texto: str) -> bool:
    palabras = [
        "mantenimiento", "orden", "repuesto",
        "falla", "equipo", "registro",
        "inspección", "trabajo"
    ]
    return any(p in texto.lower() for p in palabras)

# ===============================
# ENDPOINT CHAT
# ===============================

@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    texto: str = Form(...),
    archivo: Optional[UploadFile] = File(None)
):
    try:
        texto_documento = ""
        contexto_sheet = ""

        # 1️⃣ DOCUMENTO
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
                image = Image.open(BytesIO(data))
                texto_documento = pytesseract.image_to_string(image)

            else:
                raise HTTPException(status_code=400, detail="Tipo no soportado")

        # 2️⃣ GOOGLE SHEET
        if not texto_documento and requiere_sheet(texto):
            contexto_sheet = obtener_mantenimientos_google_sheet(texto)

        # 3️⃣ MEMORIA CORTA
        historial = memoria_conversacion[session_id]
        contexto_memoria = ""

        for h in historial[-3:]:  # solo últimas 3
            contexto_memoria += f"\nUsuario: {h['usuario']}\nAsistente: {h['asistente']}\n"

        # 4️⃣ MOTOR HÍBRIDO
        if contexto_sheet:
            prompt = f"""
CONVERSACIÓN PREVIA:
{contexto_memoria}

REGISTROS REALES:
{contexto_sheet}

Pregunta:
{texto}
"""
        elif texto_documento:
            prompt = f"""
CONVERSACIÓN PREVIA:
{contexto_memoria}

DOCUMENTO:
{texto_documento}

Pregunta:
{texto}
"""
        else:
            prompt = f"""
CONVERSACIÓN PREVIA:
{contexto_memoria}

Pregunta:
{texto}
"""

        # 5️⃣ RESPUESTA GEMINI
        response = model.generate_content(prompt)

        respuesta_texto = response.text

        # Guardar memoria
        memoria_conversacion[session_id].append({
            "usuario": texto,
            "asistente": respuesta_texto
        })

        return {"respuesta": respuesta_texto}

    except Exception as e:
        print("ERROR:", e)
        return {"respuesta": "Error procesando la solicitud."}

# ===============================
# HOME
# ===============================

@app.get("/")
def home():
    return {"status": "Servidor IA Empresarial Optimizado Activo"}