import google.generativeai as genai
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import os
import pandas as pd
import io
import requests
import unicodedata
import re
import time
import pdfplumber
from docx import Document


STOPWORDS = [
    "que", "hubo", "en", "la", "el", "los", "las",
    "de", "del", "al", "y", "o", "un", "una",
    "hay", "existe", "trabajo", "trabajos",
    "línea", "linea", "por", "para", "con"
]

def limpiar_query(query):
    palabras = query.lower().split()
    palabras_filtradas = [p for p in palabras if p not in STOPWORDS]
    return normalizar(" ".join(palabras_filtradas))

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
cache_excel = {
    "df": None,
    "last_update": 0,
    "vectorizer": None,
    "tfidf_matrix": None
}

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

def buscar_en_sheet(query):
    global cache_excel

    try:
        # ==========================================
        # 1️⃣ ACTUALIZAR CACHE CADA 5 MINUTOS
        # ==========================================
        if cache_excel["df"] is None or (time.time() - cache_excel["last_update"]) > 300:

            res = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)

            df_raw = pd.read_csv(
                io.BytesIO(res.content),
                encoding="utf-8",
                sep=None,
                engine="python"
            ).fillna("")

            # Formatear fecha si existe
            col_fecha = "FECHA (DÍA 01)"
            if col_fecha in df_raw.columns:
                df_raw[col_fecha] = pd.to_datetime(
                    df_raw[col_fecha], errors='coerce'
                ).dt.strftime('%d-%m-%Y')

            df_raw = df_raw.astype(str).replace(r'\.0$', '', regex=True)

            # ==========================================
            # 2️⃣ CREAR COLUMNA DE BÚSQUEDA UNIFICADA
            # ==========================================
            df_raw["busqueda"] = (
                df_raw.astype(str)
                .agg(' '.join, axis=1)
                .apply(normalizar)
            )

            # ==========================================
            # 3️⃣ CREAR MODELO TF-IDF
            # ==========================================
            vectorizer = TfidfVectorizer(
            ngram_range=(1,2),   # detecta frases como "linea belando"
            min_df=1,
            sublinear_tf=True)
            tfidf_matrix = vectorizer.fit_transform(df_raw["busqueda"])

            # Guardar en cache
            cache_excel["df"] = df_raw
            cache_excel["vectorizer"] = vectorizer
            cache_excel["tfidf_matrix"] = tfidf_matrix
            cache_excel["last_update"] = time.time()

        # ==========================================
        # 4️⃣ RECUPERAR DEL CACHE
        # ==========================================
        df = cache_excel["df"]
        vectorizer = cache_excel["vectorizer"]
        tfidf_matrix = cache_excel["tfidf_matrix"]

                # ==========================================
        # 🔢 DETECTAR BÚSQUEDA NUMÉRICA (ORDEN)
        # ==========================================

        query_solo_num = re.sub(r'[^0-9]', '', str(query))

        if len(query_solo_num) >= 4:  # mínimo 4 dígitos para evitar ruido

            mask_numerica = df.astype(str).apply(
                lambda col: col.str.contains(query_solo_num, na=False)
            )

            coincidencias = df[mask_numerica.any(axis=1)]

            if not coincidencias.empty:

                columnas_importantes = [
                    "N° DE ORDEN",
                    "FECHA (DÍA 01)",
                    "DESCRIPCIÓN DEL TRABAJO",
                    "STATUS 1",
                    "DIA 1) TEC. N° 01"
                ]

                columnas_validas = [
                    c for c in columnas_importantes if c in coincidencias.columns
                ]

                return coincidencias[columnas_validas].head(15).to_markdown(index=False)

        # ==========================================
        # 5️⃣ LIMPIAR CONSULTA
        # ==========================================
        query_limpia = limpiar_query(query)
        

        if not query_limpia.strip():
            return ""

        query_normalizada = normalizar(query_limpia)

        

        # ==========================================
        # 6️⃣ TRANSFORMAR QUERY A VECTOR
        # ==========================================
        query_vec = vectorizer.transform([query_normalizada])

        similitudes = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # ==========================================
        # 7️⃣ OBTENER TOP 15 RESULTADOS
        # ==========================================
        top_indices = similitudes.argsort()[-15:][::-1]

        resultado = df.iloc[top_indices]

        # Filtrar solo si similitud > 0
        threshold = 0.1
        resultado = resultado[similitudes[top_indices] > threshold]

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
        print("Error búsqueda TF-IDF:", e)
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