import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pdfplumber
from docx import Document
from io import BytesIO

def extraer_texto_pdf(data: bytes) -> str:
    texto = ""
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            texto += page.extract_text() or ""
    return texto

def extraer_texto_docx(data: bytes) -> str:
    doc = Document(BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)

# 1. Configura tu IA (Asegúrate de poner tu API Key real aquí)
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name='gemini-2.5-flash-lite',

generation_config={"temperature": 0.7},
system_instruction=
        """
Eres un asistente general en español.
Puedes responder preguntas, explicar conceptos y analizar documentos.

REGLAS:
- Responde siempre en ESPAÑOL.
- Si el usuario adjunta un documento, úsalo como CONTEXTO para responder.
- No resumas documentos automáticamente, a menos que el usuario lo solicite.
- Si la pregunta se refiere al contenido del documento, apóyate en él.
-si no hay documento, responde correctamente la pregunta que te realicen
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
        texto_documento = ""

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
                # IMÁGENES: sí se envían directo
                contenido = [
                    texto,
                    {"mime_type": mime, "data": data}
                ]
                response = model.generate_content(contenido)
                return {"respuesta": response.text}

            else:
                raise HTTPException(status_code=400, detail="Tipo de archivo no soportado")

        # DOCUMENTOS → TEXTO
        prompt = f"""
        {texto}

        CONTENIDO DEL DOCUMENTO:
        {texto_documento[:8000]}
        """

        response = model.generate_content(prompt)

        return {"respuesta": response.text}

    except Exception as e:
        print("ERROR:", e)
        return {"respuesta": "No pude procesar el documento. Puede estar escaneado o dañado."}

@app.get("/")
def home():
    return {"status": "Servidor de IA Activo"}