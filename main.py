import google.generativeai as genai
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
import os
import pandas as pd
import io
import requests
import unicodedata
import re
import time
import pdfplumber
from docx import Document

# ==========================================
# NORMALIZADOR
# ==========================================

def normalizar(texto):
    if not texto:
        return ""
    texto = str(texto).lower()
    texto = "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return texto.strip()

# ==========================================
# CONFIGURACIÓN GEMINI
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 40,
        "response_mime_type": "text/plain",
    },
    system_instruction=(
        "Eres un asistente técnico experto en gestión de mantenimiento. "
        "Responde únicamente en español. "
        "NO debes inventar información. "
        "Si no hay datos en 'Registros Excel', debes responder exactamente: "
        "'No existe información en el registro para esta consulta.'"
    )
)

# ==========================================
# FASTAPI
# ==========================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

memoria_conversacion = defaultdict(list)
cache_excel = {"df": None, "last_update": 0}

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1oEVKH1SxHDJtwSx9y3sy1Ui12CqvCWdRTb9bEe_w4D8/export?format=csv&gid=1960130423"

# ==========================================
# EXTRACTORES LOCALES
# ==========================================

def extraer_de_pdf(bytes_file):
    with pdfplumber.open(io.BytesIO(bytes_file)) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def extraer_de_docx(bytes_file):
    doc = Document(io.BytesIO(bytes_file))
    return "\n".join([p.text for p in doc.paragraphs])

def extraer_de_excel_adjunto(bytes_file):
    df = pd.read_excel(io.BytesIO(bytes_file))
    return df.head(20).to_markdown(index=False)

# ==========================================
# BÚSQUEDA INTELIGENTE EN GOOGLE SHEET
# ==========================================

# ==========================================
# BÚSQUEDA INTELIGENTE EN GOOGLE SHEET
# ==========================================

def buscar_en_sheet(query):
    global cache_excel
    try:
        # Cache 5 minutos
        if cache_excel["df"] is None or (time.time() - cache_excel["last_update"]) > 300:
            res = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)
            df_raw = pd.read_csv(
                io.BytesIO(res.content),
                encoding="utf-8",
                sep=None,
                engine="python"
            ).fillna("")

            col_fecha = "FECHA (DÍA 01)"
            if col_fecha in df_raw.columns:
                df_raw[col_fecha] = pd.to_datetime(
                    df_raw[col_fecha], errors='coerce'
                ).dt.strftime('%d-%m-%Y')

            cache_excel["df"] = df_raw.astype(str)
            cache_excel["last_update"] = time.time()

        df = cache_excel["df"].copy()

        print("Columnas detectadas:", df.columns.tolist())
        print("Primeras 3 filas:")
        print(df.head(3))
           

        query_normalizada = normalizar(query)

        if not query_normalizada:
            return ""

        # =====================================================
        # 🔎 SI LA CONSULTA ES PRINCIPALMENTE NUMÉRICA
        # =====================================================
        if query_normalizada.isdigit():

            # 1️⃣ Buscar coincidencia EXACTA en cualquier columna
            exactos = df[
                df.apply(
                    lambda fila: any(
                        str(valor).strip().replace(".0", "") == query_normalizada
                        for valor in fila
                    ),
                    axis=1
                )
            ]

            if not exactos.empty:
                resultado = exactos
            else:
                # 2️⃣ Buscar coincidencias que EMPIECEN por ese número
                parciales = df(
                    df.apply(
                        lambda fila: any(
                            str(valor).strip().replace(".0", "").startswith(query_normalizada)
                            for valor in fila
                        ),
                        axis=1
                    ),
                    axis=1
                )

                resultado = parciales

        else:
            # =====================================================
            # 🔎 BÚSQUEDA NORMAL POR TEXTO (TU LÓGICA MEJORADA)
            # =====================================================

            df["contenido_completo"] = df.apply(
                lambda x: ' '.join(x.astype(str)), axis=1
            )

            df["contenido_normalizado"] = df["contenido_completo"].apply(normalizar)

            coincidencia_directa = df[
                df["contenido_normalizado"].str.contains(query_normalizada, na=False)
            ]

            if not coincidencia_directa.empty:
                resultado = coincidencia_directa
            else:
                palabras = query_normalizada.split()

                score = df["contenido_normalizado"].apply(
                    lambda fila: sum(1 for p in palabras if p in fila)
                )

                df["score"] = score

                resultado = df[df["score"] > 0].sort_values(
                    by="score", ascending=False
                )

        if resultado.empty:
            return ""

        columnas_importantes = [
            "N° DE ORDEN",
            "FECHA (DÍA 01)",
            "DESCRIPCIÓN DEL TRABAJO",
            "STATUS 1",
            "DIA 1) TEC. N° 01"
        ]

        columnas_validas = [
            c for c in columnas_importantes if c in resultado.columns
        ]

        return resultado[columnas_validas].head(15).to_markdown(index=False)

    except Exception as e:
        print("Consulta recibida:", query)
        print("Consulta normalizada:", query_normalizada)
        print(f"Error búsqueda: {e}")
        return ""

           

# ==========================================
# ENDPOINT PRINCIPAL
# ==========================================

@app.post("/chat")
async def chat(
    texto: str = Form(None),
    session_id: str = Form("default_session"),
    archivo: UploadFile = File(None)
):
    try:

        texto_extraido = ""

        # -------- PROCESAMIENTO DE ARCHIVO --------
        if archivo:
            mimetype = archivo.content_type
            bytes_file = await archivo.read()

            if "pdf" in mimetype:
                texto_extraido = extraer_de_pdf(bytes_file)
            elif "word" in mimetype or "officedocument.wordprocessingml" in mimetype:
                texto_extraido = extraer_de_docx(bytes_file)
            elif "excel" in mimetype or "officedocument.spreadsheetml" in mimetype:
                texto_extraido = extraer_de_excel_adjunto(bytes_file)

        # -------- BÚSQUEDA EN GOOGLE SHEET --------
        contexto_sheet = buscar_en_sheet(texto or "")

        # 🔴 BLOQUEO ANTI-ALUCINACIÓN
        if not contexto_sheet:
            return {
                "respuesta": "No existe información en el registro para esta consulta.",
                "tokens_usados": 0
            }

        historial = memoria_conversacion[session_id]
        historial_txt = "\n".join(
            [f"U: {h['u']}\nA: {h['a']}" for h in historial[-2:]]
        )

        # -------- PROMPT RESTRICTIVO --------
        prompt = f"""
Responde únicamente utilizando la información exacta contenida en la sección 'Registros Excel'.
No agregues información externa.
Si algo no está explícitamente en los registros, responde:
"No existe información en el registro para esta consulta."

Historial:
{historial_txt}

Registros Excel:
{contexto_sheet}

Usuario: {texto}
"""

        response = model.generate_content(prompt)

        usage = response.usage_metadata

        print(f"\n--- REPORTE DE CONSUMO (Sesión: {session_id}) ---")
        print(f"Tokens Entrada: {usage.prompt_token_count}")
        print(f"Tokens Salida: {usage.candidates_token_count}")
        print(f"Tokens Totales: {usage.total_token_count}")
        print("------------------------------------------\n")

        memoria_conversacion[session_id].append({
            "u": texto,
            "a": response.text
        })

        if len(memoria_conversacion[session_id]) > 5:
            memoria_conversacion[session_id].pop(0)

        return {
            "respuesta": response.text,
            "tokens_usados": usage.total_token_count
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"respuesta": "Error procesando la solicitud."}

# ==========================================
# ENDPOINT STATUS
# ==========================================

@app.get("/")
def home():
    return {"status": "Servidor IA Activo"}