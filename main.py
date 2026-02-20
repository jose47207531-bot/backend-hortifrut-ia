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
# CONFIGURACIÓN
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", # Más económico para tareas empresariales
    generation_config={"temperature": 0.2},
    system_instruction="Eres un asistente de mantenimiento. Usa el contexto de los archivos y el Excel para responder de forma técnica y breve."
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

memoria_conversacion = defaultdict(list)
cache_excel = {"df": None, "last_update": 0}
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1oEVKH1SxHDJtwSx9y3sy1Ui12CqvCWdRTb9bEe_w4D8/export?format=csv&gid=1960130423"

# ==========================================
# EXTRACTORES LOCALES (Para ahorrar tokens)
# ==========================================

def extraer_de_pdf(bytes_file):
    with pdfplumber.open(io.BytesIO(bytes_file)) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def extraer_de_docx(bytes_file):
    doc = Document(io.BytesIO(bytes_file))
    return "\n".join([p.text for p in doc.paragraphs])

def extraer_de_excel_adjunto(bytes_file):
    df = pd.read_excel(io.BytesIO(bytes_file))
    return df.head(20).to_markdown(index=False) # Solo enviamos una muestra para ahorrar

# ==========================================
# LÓGICA DE BÚSQUEDA EN GOOGLE SHEET
# ==========================================

def normalizar(t):
    if not t: return ""
    return unicodedata.normalize("NFD", t.lower()).encode("ascii", "ignore").decode("utf-8")

def buscar_en_sheet(query, historial=""):
    global cache_excel
    try:
        if cache_excel["df"] is None or (time.time() - cache_excel["last_update"]) > 300:
            res = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)
            cache_excel["df"] = pd.read_csv(io.StringIO(res.text)).fillna("")
            cache_excel["last_update"] = time.time()
        
        df = cache_excel["df"]
        terminos = [p for p in normalizar(f"{query} {historial}").split() if len(p) > 2]
        
        if not terminos: return ""
        
        # Búsqueda eficiente en todas las celdas
        mask = df.apply(lambda row: any(t in normalizar(" ".join(row.astype(str))) for t in terminos), axis=1)
        res_df = df[mask].head(5)
        
        columnas = [c for c in df.columns if any(k in c.lower() for k in ["orden", "desc", "obs", "resp", "equipo", "estado"])]
        return res_df[columnas].to_markdown(index=False) if not res_df.empty else ""
    except: return ""

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
        mimetype = ""
        
        # 1. Procesar archivo localmente para no gastar tokens de imagen/multimedia si es texto
        if archivo:
            mimetype = archivo.content_type
            bytes_file = await archivo.read()
            
            if "pdf" in mimetype:
                texto_extraido = f"\n[Contenido del PDF adjunto]:\n{extraer_de_pdf(bytes_file)}"
            elif "word" in mimetype or "officedocument.wordprocessingml" in mimetype:
                texto_extraido = f"\n[Contenido del Word adjunto]:\n{extraer_de_docx(bytes_file)}"
            elif "excel" in mimetype or "officedocument.spreadsheetml" in mimetype:
                texto_extraido = f"\n[Contenido del Excel adjunto]:\n{extraer_de_excel_adjunto(bytes_file)}"
            elif "image" in mimetype or "video" in mimetype:
                # Si es imagen o video, no hay de otra: enviamos a Gemini para que use su visión
                # Pero esto solo pasará con archivos visuales
                input_data = [{"mime_type": mimetype, "data": bytes_file}, texto or "Analiza esto"]
                response = model.generate_content(input_data)
                return {"respuesta": response.text}

        # 2. Si no fue imagen/video, procesamos como texto (más barato)
        historial = memoria_conversacion[session_id]
        historial_txt = "\n".join([f"U: {h['u']}\nA: {h['a']}" for h in historial[-2:]])
        
        # 3. Búsqueda en Google Sheet
        contexto_sheet = buscar_en_sheet(texto or "", historial_txt)
        
        # 4. Construir Prompt de texto puro (Ahorro máximo de tokens)
        prompt = f"Historial:\n{historial_txt}\n"
        if contexto_sheet: prompt += f"\nRegistros Excel:\n{contexto_sheet}\n"
        if texto_extraido: prompt += f"\nDocumento adjunto:\n{texto_extraido}\n"
        prompt += f"\nUsuario: {texto or 'Analiza el documento'}"

        response = model.generate_content(prompt)
        
        # Guardar memoria corta
        memoria_conversacion[session_id].append({"u": texto, "a": response.text})
        if len(memoria_conversacion[session_id]) > 5: memoria_conversacion[session_id].pop(0)

        return {"respuesta": response.text}

    except Exception as e:
        print(f"Error: {e}")
        return {"respuesta": "Error procesando la solicitud."}

@app.get("/")
def home(): return {"status": "Servidor IA Activo"}