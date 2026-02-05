import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

# 1. Configura tu IA (Asegúrate de poner tu API Key real aquí)
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
generation_config={"temperature": 0.7},
system_instruction=(
        """
    Eres un asistente de oficina que habla estrictamente en ESPAÑOL. 
    Tu tarea es analizar archivos Word, Excel, PDF e imágenes.
    REGLA DE ORO: Todas tus respuestas deben ser en ESPAÑOL, sin excepciones. 
    Si el usuario te envía un documento, léelo y resume su contenido en castellano.
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



@app.post("/chat")
async def chat(texto: str = Form(...), archivo: Optional[UploadFile] = File(None)):
    try:
        # 1. Leemos los bytes primero
        if archivo:
            doc_data = await archivo.read()
            # 2. Creamos la lista de partes para Gemini
            # Pasamos el mime_type real que detecta FastAPI
            contenido = [
                texto, 
                {"mime_type": archivo.content_type, "data": doc_data}
            ]
        else:
            contenido = [texto]

        # 3. Llamada al modelo
        response = model.generate_content(contenido)
        
        if not response.text:
            return {"respuesta": "La IA recibió el archivo pero no pudo generar texto. Intenta con un archivo más pequeño."}
            
        return {"respuesta": response.text}
    except Exception as e:
        print(f"Error: {e}")
        return {"respuesta": "Error procesando el documento. Asegúrate de que no esté protegido con contraseña."}

@app.get("/")
def home():
    return {"status": "Servidor de IA Activo"}