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
from core.analytics import ejecutar_analisis, generar_analisis_tecnico_avanzado
from core.rag import buscar_en_sheet, obtener_dataframe
from core.rag import normalizar


def es_consulta_tecnica(texto):
    if not texto:
        return False

    texto = texto.lower()

    palabras_tecnicas = [
        "orden", "trabajo", "tecnico", "técnico",
        "fecha", "equipo", "status", "linea",
        "línea", "mantenimiento", "falla",
        "registro", "intervino", "stock", "repuestos", "trabajador"
    ]

    # Detectar número largo (posible número de orden)
    if re.search(r'\d{4,}', texto):
        return True

    # Detectar palabra técnica
    for palabra in palabras_tecnicas:
        if palabra in texto:
            return True

    return False

def es_pregunta_analitica(texto):
    if not texto:
        return False

    texto = texto.lower()

    palabras = [
        "recomienda",
        "recomendacion",
        "evitar",
        "prevenir",
        "mejorar",
        "analiza",
        "analizar",
        "tendencia",
        "patron",
        "patrón",
        "indicador",
        "optimizar"
    ]

    return any(p in texto for p in palabras)

# ==========================================
# ANALISIS DINAMICO DE ENTIDAD
# ==========================================

def analizar_entidad(df, texto_usuario):
    if df is None or texto_usuario is None:
        return None

    texto_norm = normalizar(texto_usuario)

    mask = df.astype(str).apply(
        lambda col: col.str.contains(texto_norm, case=False, na=False)
    )

    df_filtrado = df[mask.any(axis=1)]

    if df_filtrado.empty:
        return None

    total_registros = len(df_filtrado)

    falla_frecuente = None
    if "DESCRIPCIÓN DEL TRABAJO" in df_filtrado.columns:
        falla_frecuente = (
            df_filtrado["DESCRIPCIÓN DEL TRABAJO"]
            .value_counts()
            .idxmax()
        )

    return {
        "total": total_registros,
        "falla_frecuente": falla_frecuente
    }


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
       "Eres un ingeniero senior de mantenimiento experto en gestión de mantenimiento industrial. "
    "Responde siempre en español. "
    "Cuando el usuario consulte sobre órdenes, equipos, técnicos, fechas o trabajos, "
    "solo puedes usar la información contenida en 'Registros Excel'. "
    "Si la información no está en los registros, debes responder exactamente: "
    "'No existe información en el registro para esta consulta.' "
    "Si el usuario hace preguntas generales, saludos o pide recomendaciones técnicas generales, "
    "puedes responder normalmente sin depender de los registros."
    )
)

# ==========================================
# FASTAPI
# ==========================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

memoria_conversacion = defaultdict(list)
# 🔵 NUEVA MEMORIA PARA CONTEXTO DEL EXCEL
memoria_contexto_sheet = {}
cache_excel = {
    "df": None,
    "last_update": 0,
    "vectorizer": None,
    "tfidf_matrix": None
}

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/12z2M2H_iE6MAKjgPbDwmt2HaJ7ZQRfx_PL0jDxbQnS8/edit?gid=955581654#gid=955581654"

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
        # Si se extrajo texto del archivo, lo agregamos a la consulta
        if texto_extraido:
         texto = (texto or "") + "\n\nContenido del archivo:\n" + texto_extraido

        # -------- CLASIFICADOR ANTES DEL RAG --------
        usar_excel = es_consulta_tecnica(texto)
        es_analitica = es_pregunta_analitica(texto)
        df = obtener_dataframe()
        # ==============================
        # 🔥 MODO ANALÍTICO AVANZADO
        # ==============================
        analisis = None

        if es_analitica and df is not None:
         analisis = ejecutar_analisis(df, texto)
         analisis_tecnico = generar_analisis_tecnico_avanzado(df, texto)

        if analisis is not None:

         resumen = f"Resultado analítico detectado: {analisis}"

         prompt = f"""
          Eres un ingeniero senior de mantenimiento experto en una planta industrial.

          Tu objetivo NO es solo responder, sino ayudar a tomar decisiones operativas.

          Tienes dos fuentes de información:

         1) Datos analíticos:
         {resumen}

         2) Historial real:
         {analisis_tecnico}

         Debes generar una respuesta estructurada con:

         🔍 Hallazgos:
         - Qué está ocurriendo
         - Qué patrones detectas

         ⚠️ Riesgos:
         - Qué podría fallar si continúa la tendencia

         🛠️ Recomendaciones:
         - Acciones concretas
         - Tipo de mantenimiento (preventivo, correctivo, predictivo)

         📊 Insight técnico:
         - Interpretación profesional (como ingeniero)

         Usuario: {texto}
          """

         response = model.generate_content(prompt)
         usage = response.usage_metadata

        else:
       
         if usar_excel:
          contexto_sheet = buscar_en_sheet(texto or "")
         else:
          contexto_sheet = ""

# 🔵 Si encontró algo nuevo, lo guardamos
        if contexto_sheet:
           memoria_contexto_sheet[session_id] = contexto_sheet
        
    # 🔵 Si no encontró nada pero hay contexto previo, lo reutilizamos
        elif session_id in memoria_contexto_sheet:
            contexto_sheet = memoria_contexto_sheet[session_id]

        # 🔴 BLOQUEO ANTI-ALUCINACIÓN
        #if not contexto_sheet:return {"respuesta": "No existe información en el registro para esta consulta.","tokens_usados": 0}

        historial = memoria_conversacion[session_id]
        historial_txt = "\n".join(
            [f"U: {h['u']}\nA: {h['a']}" for h in historial[-2:]]
        )

        # -------- PROMPT RESTRICTIVO --------

        if contexto_sheet:
         prompt = f"""
Eres un ingeniero senior de mantenimiento en gestión de mantenimiento industrial.

Debes usar EXCLUSIVAMENTE la información contenida en 'Registros Excel'.

Puedes:
- Analizar los datos
- Calcular indicadores
- Hacer conteos
- Identificar patrones
- Generar métricas
- Resumir información

NO puedes:
- Inventar órdenes
- Inventar técnicos
- Inventar fechas
- Agregar datos que no estén en los registros

Debes interpretar y responder como un experto en el tema.
Si la información necesaria no está en los registros,
debes responder exactamente:
"No existe información en la base de datos para esta consulta, puedes detallarme tu pregunta o puedes realizar otra."

Historial:
{historial_txt}

Registros Excel:
{contexto_sheet}

Usuario: {texto}
"""
        else:
         prompt = f"""
         Responde como asistente técnico experto en mantenimiento industrial.
         Si la pregunta es general, responde normalmente.
         No inventes datos específicos de órdenes o registros.

         Historial:
         {historial_txt}

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