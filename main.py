import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Configura tu IA (Asegúrate de poner tu API Key real aquí)
genai.configure(api_key="AIzaSyBJdw-4qtQbBYIkcjKIH_rJ938eeJOutdc")
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# Permisos para que no se bloquee el navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pregunta(BaseModel):
    texto: str

@app.post("/chat")
async def chat(pregunta: Pregunta):
    try:
        # Esto envía la pregunta directamente a la IA de Google
        response = model.generate_content(pregunta.texto)
        return {"respuesta": response.text}
    except Exception as e:
        return {"error": f"Error de conexión: {str(e)}"}

@app.get("/")
def home():
    return {"status": "Servidor de IA Activo"}
