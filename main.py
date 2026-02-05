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
model = genai.GenerativeModel('gemini-2.5-flash-lite')
generation_config={"temperature": 0.7},
system_instruction=(
        "Eres un asistente que habla ÚNICAMENTE en español. "
        "Si recibes una imagen o archivo adjunto como PDF, word, video, analízalo a fondo y responde "
        "directamente sobre su contenido en español."
        "leerlo si el archivo está adjunto."
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
        contenido_para_gemini = [texto]

        # Si el usuario envió un archivo (Imagen o PDF)
        if archivo:
            bytes_data = await archivo.read()
            contenido_para_gemini.append({
                "mime_type": archivo.content_type,
                "data": bytes_data
            })

        # Enviamos el texto y el archivo juntos a la IA
        response = model.generate_content(contenido_para_gemini)
        
        return {"respuesta": response.text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")

@app.get("/")
def home():
    return {"status": "Servidor de IA Activo"}