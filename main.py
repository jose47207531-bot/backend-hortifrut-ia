import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
import requests
from typing import Optional
import pdfplumber
import docx
from PIL import Image
import pytesseract
import io

# ===============================
# CONFIG GEMINI
# ===============================

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    generation_config={"temperature": 0.3},
    system_instruction="""
Eres un asistente empresarial experto en mantenimiento industrial.

Reglas:
- Responde siempre en español.
- Usa los datos estructurados cuando existan.
- Si no hay datos suficientes, dilo claramente.
- Si es pregunta gerencial, responde con enfoque ejecutivo.
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
# CONFIG GOOGLE SHEETS
# ===============================

SPREADSHEET_ID = "1oEVKH1SxHDJtwSx9y3sy1Ui12CqvCWdRTb9bEe_w4D8"

SHEETS = {
    "OM GENERAL": 628280434,
    "DATOS COMPARTIDOS": 1885887150,
    "OMAISLAMIENTO": 743572316,
    "OMAIRESACONDICIONADOS": 396018642,
    "OMPAT": 328411278,
    "OMVNH3": 1394368478,
    "OMCOMEDORES": 1186193865,
    "OMGEE": 1304531883,
    "OMIRM": 190759305,
    "OMTRANSPALETA1": 580356100,
}

def cargar_sheet(nombre):
    gid = SHEETS[nombre]
    url = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/export?format=csv&gid={gid}"
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        return df
    except:
        return pd.DataFrame()

# ===============================
# INTENCIONES EMPRESARIALES
# ===============================

PALABRAS_CONTEO = ["cuántos", "cuantos", "cantidad", "total", "numero"]
PALABRAS_LISTAR = ["mostrar", "listar", "detalle", "ver"]
PALABRAS_ULTIMO = ["último", "ultima", "reciente"]
PALABRAS_GERENCIA = [
    "kpi", "indicador", "eficiencia",
    "retrabajo", "tiempo promedio",
    "horas totales", "estadística"
]

def detectar_intencion(texto):
    texto = texto.lower()

    if any(p in texto for p in PALABRAS_GERENCIA):
        return "gerencia"
    if any(p in texto for p in PALABRAS_CONTEO):
        return "conteo"
    if any(p in texto for p in PALABRAS_LISTAR):
        return "listar"
    if any(p in texto for p in PALABRAS_ULTIMO):
        return "ultimo"

    return "desconocido"

def detectar_sheet(texto):
    texto = texto.lower()
    for nombre in SHEETS.keys():
        if nombre.lower().replace("om", "") in texto:
            return nombre
    if "nh3" in texto:
        return "OMVNH3"
    if "transpaleta" in texto:
        return "OMTRANSPALETA1"
    return None

def detectar_filtros(texto, df):
    texto = texto.lower()
    filtros = {}

    for columna in df.columns:
        valores = df[columna].astype(str).str.lower().unique()
        for v in valores:
            if v and len(v) > 3 and v in texto:
                filtros[columna] = v

    return filtros

# ===============================
# MOTOR DE REGLAS AVANZADO
# ===============================

def motor_reglas(texto):

    sheet_detectado = detectar_sheet(texto)

    if not sheet_detectado:
        return None

    df = cargar_sheet(sheet_detectado)

    if df.empty:
        return "No se pudo cargar información de la base."

    intencion = detectar_intencion(texto)
    filtros = detectar_filtros(texto, df)

    df_filtrado = df.copy()

    for col, val in filtros.items():
        df_filtrado = df_filtrado[df_filtrado[col].astype(str).str.lower() == val]

    if df_filtrado.empty:
        return "No se encontraron registros con esos criterios."

    # ================= GERENCIA =================
    if intencion == "gerencia":

        total = len(df)
        retrabajos = 0

        if "RE-TRABAJO?" in df.columns:
            retrabajos = df[df["RE-TRABAJO?"].astype(str).str.upper() == "SI"].shape[0]

        horas = 0
        if "TIEMPO TOTAL DE EJECUCIÓN" in df.columns:
            horas = pd.to_numeric(df["TIEMPO TOTAL DE EJECUCIÓN"], errors="coerce").sum()

        return f"""
📊 REPORTE EJECUTIVO – {sheet_detectado}

Total Órdenes: {total}
Retrabajos: {retrabajos}
Horas Totales Ejecutadas: {round(horas,2)}

Nivel de Retrabajo: {round((retrabajos/total)*100,2) if total>0 else 0} %
"""

    # ================= CONTEO =================
    if intencion == "conteo":
        return f"Hay {len(df_filtrado)} registros en {sheet_detectado} que cumplen el criterio."

    # ================= ULTIMO =================
    if intencion == "ultimo":
        if "FECHA PROGRAMADA" in df.columns:
            df_filtrado["FECHA PROGRAMADA"] = pd.to_datetime(df_filtrado["FECHA PROGRAMADA"], errors="coerce")
            df_filtrado = df_filtrado.sort_values("FECHA PROGRAMADA", ascending=False)
        return df_filtrado.head(1).to_string(index=False)

    # ================= LISTAR =================
    if intencion == "listar":
        return df_filtrado.head(10).to_string(index=False)

    return df_filtrado.head(5).to_string(index=False)

# ===============================
# ENDPOINT PRINCIPAL
# ===============================

@app.post("/chat")
async def chat(texto: str = Form(...)):

    # 1️⃣ Intentar reglas
    resultado = motor_reglas(texto)

    if resultado:
        return {"respuesta": resultado}

    # 2️⃣ Fallback IA con contexto multi-sheet
    contexto_total = ""

    for nombre in SHEETS.keys():
        df = cargar_sheet(nombre)
        if not df.empty:
            contexto_total += f"\n\n=== {nombre} ===\n"
            contexto_total += df.head(20).to_string(index=False)

    prompt = f"""
Pregunta del usuario:
{texto}

Datos disponibles:
{contexto_total}
"""

    try:
        response = model.generate_content(prompt)
        return {"respuesta": response.text}
    except Exception as e:
        print("ERROR:", e)
        return {"respuesta": "Error procesando la solicitud."}

# ===============================
# LECTURA DE ARCHIVOS
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
