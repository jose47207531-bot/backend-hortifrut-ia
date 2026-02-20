import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Request
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
import unicodedata
import re

# ==========================================
# CONFIGURACIÓN GEMINI
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ No se encontró GEMINI_API_KEY en Render")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    generation_config={"temperature": 0.4},
    system_instruction="""
Eres un asistente empresarial especializado en mantenimiento industrial.

Reglas:
- Responde siempre en español.
- Usa datos estructurados si existen.
- Si no hay datos suficientes, dilo claramente.
- Sé claro y profesional.
"""
)

# ==========================================
# FASTAPI
# ==========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# MEMORIA CONVERSACIONAL
# ==========================================

memoria_conversacion = defaultdict(list)

# ==========================================
# GOOGLE SHEET
# ==========================================

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1oEVKH1SxHDJtwSx9y3sy1Ui12CqvCWdRTb9bEe_w4D8/export?format=csv&gid=1960130423"

def normalizar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return texto

def buscar_texto_libre(df, consulta):
    consulta_norm = normalizar_texto(consulta)

    # eliminar palabras irrelevantes
    stopwords = [
        "el","la","los","las","un","una","unos","unas",
        "de","del","en","y","o","a","que","hubo","hay",
        "alguno","alguna","algun","trabajo","linea","lineas"
    ]

    palabras = [
        p for p in consulta_norm.split()
        if p not in stopwords and len(p) > 2
    ]

    if not palabras:
        return pd.DataFrame()

    def fila_coincide(fila):
        texto_fila = normalizar_texto(" ".join(fila.astype(str)))
        coincidencias = sum(1 for palabra in palabras if palabra in texto_fila)
        return coincidencias >= 1  # con 1 coincidencia ya es válido

    filtro = df.apply(fila_coincide, axis=1)
    return df[filtro]

def obtener_datos_sheet(busqueda: str = "") -> str:
    try:
        response = requests.get(GOOGLE_SHEET_CSV_URL)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))

        if df.empty:
            return ""

        df_filtrado = buscar_texto_libre(df, busqueda)

        if df_filtrado.empty:
            return ""

        columnas_clave = [
            col for col in df.columns
            if any(k in col.lower() for k in [
                "orden", "descripcion", "observacion",
                "responsable", "fecha", "trabajo",
                "equipo", "estado"
            ])
        ]

        if columnas_clave:
            df_filtrado = df_filtrado[columnas_clave]

        return df_filtrado.head(5).to_string(index=False)

    except Exception as e:
        print("❌ Error leyendo Google Sheet:", e)
        return ""

def requiere_sheet(texto: str) -> bool:
    palabras = [
        "mantenimiento", "orden", "repuesto",
        "falla", "equipo", "registro",
        "inspección", "trabajo", "estado"
    ]
    return any(p in texto.lower() for p in palabras)

# ==========================================
# DOCUMENTOS
# ==========================================

def extraer_texto_pdf(data: bytes) -> str:
    texto = ""
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            texto += page.extract_text() or ""
    return texto

def extraer_texto_docx(data: bytes) -> str:
    doc = Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)

# ==========================================
# ENDPOINT CHAT UNIVERSAL
# ==========================================

@app.post("/chat")
async def chat(request: Request):

    try:
        content_type = request.headers.get("content-type", "")
        session_id = "default_session"
        texto = None
        archivo = None

        # JSON
        if content_type.startswith("application/json"):
            body = await request.json()
            session_id = body.get("session_id", "default_session")
            texto = body.get("texto") or body.get("mensaje") or body.get("pregunta")

        # FORM DATA
        elif "multipart/form-data" in content_type:
            form = await request.form()
            session_id = form.get("session_id", "default_session")
            texto = form.get("texto") or form.get("mensaje") or form.get("pregunta")
            archivo = form.get("archivo")

        if not texto:
            return {"respuesta": "No se recibió ningún mensaje válido."}

        print("📩 Texto recibido:", texto)

        texto_documento = ""
        contexto_sheet = ""

        # Google Sheet inteligente
        if requiere_sheet(texto):
            contexto_sheet = obtener_datos_sheet(texto)

        # Memoria corta
        historial = memoria_conversacion[session_id]
        contexto_memoria = ""

        for h in historial[-3:]:
            contexto_memoria += f"\nUsuario: {h['usuario']}\nAsistente: {h['asistente']}\n"

        # Construcción del prompt
        if contexto_sheet:
            prompt = f"""
CONVERSACIÓN PREVIA:
{contexto_memoria}

REGISTROS EMPRESARIALES:
{contexto_sheet}

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

        response = model.generate_content(prompt)
        respuesta_texto = response.text

        memoria_conversacion[session_id].append({
            "usuario": texto,
            "asistente": respuesta_texto
        })

        return {"respuesta": respuesta_texto}

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"respuesta": "Error procesando la solicitud."}

# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():
    return {"status": "Servidor IA Empresarial Activo"}