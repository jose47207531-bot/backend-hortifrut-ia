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

def normalizar(texto):
    if not texto:
        return ""
    # Convertir a minúsculas
    texto = str(texto).lower()
    # Eliminar tildes y acentos (ej: Evaporador -> evaporador)
    texto = "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    # Eliminar caracteres especiales excepto letras y números
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return texto.strip()

# ==========================================
# CONFIGURACIÓN
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
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

def buscar_en_sheet(query, historial=""):
    global cache_excel
    try:
        # 1. Carga con Cache (Esto ya lo haces bien)
        if cache_excel["df"] is None or (time.time() - cache_excel["last_update"]) > 300:
            res = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)
            # Forzamos que todo sea string para evitar errores de tipo
            cache_excel["df"] = pd.read_csv(io.StringIO(res.text)).fillna("").astype(str)
            cache_excel["last_update"] = time.time()
        
        df = cache_excel["df"].copy()
        
        # 2. Limpieza de la consulta
        # Eliminamos palabras vacías (de, el, la, un, para) para quedarnos con lo importante
        stop_words = ["de", "el", "la", "un", "una", "en", "para", "con", "hay", "que", "si", "sobre"]
        palabras_clave = [normalizar(w) for w in query.split() if len(w) > 2 and w.lower() not in stop_words]
        
        if not palabras_clave: return ""

        # 3. Búsqueda Multi-Palabra (Identifica TODO)
        # Buscamos filas donde aparezcan las palabras clave
        # Creamos una serie que concatena las columnas relevantes para buscar rápido
        # (Ajusta las columnas si prefieres buscar solo en 'DESCRIPCIÓN' y 'ORDEN')
        contenido_busqueda = df.apply(lambda x: ' '.join(x), axis=1).apply(normalizar)
        
        # Filtro: La fila debe contener AL MENOS una de las palabras clave principales
        # Opcional: Puedes cambiar '.any()' por '.all()' si quieres que sea más estricto
        masks = [contenido_busqueda.str.contains(p, case=False, na=False) for p in palabras_clave]
        
        # Combinamos las máscaras (esto encontrará "evaporador", "evaporadores", "evap", etc.)
        final_mask = masks[0]
        for m in masks[1:]:
            final_mask = final_mask & m # Usamos & (AND) para que busque filas que tengan ambas palabras
            
        res_df = df[final_mask]

        # 4. Control de "Peso" de Tokens
        # Si encuentra 100 filas, mandarlas todas a Gemini saldría carísimo.
        # Limitamos a las 20 más recientes (asumiendo que las últimas filas son las nuevas)
        res_df = res_df.tail(20) 
        
        if res_df.empty:
            return ""

        return res_df.to_markdown(index=False)
        
    except Exception as e:
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