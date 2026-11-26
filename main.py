import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from core.insights import obtener_insights, obtener_columna_principal
import os
import pandas as pd
import io
import re
import threading
import pdfplumber
from docx import Document
from core.analytics import ejecutar_analisis, generar_analisis_tecnico_avanzado
from core.rag import buscar_en_sheet, obtener_dataframe, formatear_contexto, normalizar, cargar_datos
from PIL import Image

# Carga el Sheet en segundo plano para que el servidor arranque rápido
threading.Thread(target=cargar_datos, daemon=True).start()

memoria_usuario = {
    "ultimo_equipo": None,
    "ultimo_resultado": None
}


def es_consulta_tecnica(texto):
    if not texto:
        return False

    texto = texto.lower()

    palabras_tecnicas = [
        "orden", "trabajo", "tecnico", "técnico",
        "fecha", "equipo", "status", "linea",
        "línea", "mantenimiento", "falla",
        "registro", "intervino", "stock",
        "repuestos", "trabajador",
        "paso", "pasó", "historial", "ocurrio",
        "ocurrió"
    ]

    # 🔥 Detectar códigos tipo HO-233-EVNH3
    if re.search(r'[a-z]{2,}-?\d+', texto):
        return True

    # Detectar números largos (orden)
    if re.search(r'\d{4,}', texto):
        return True

    return any(p in texto for p in palabras_tecnicas)


def es_pregunta_analitica(texto):
    if not texto:
        return False

    texto = texto.lower()

    palabras = [
        "recomienda", "recomendacion", "evitar", "prevenir", "mejorar",
        "analiza", "analizar", "tendencia", "patron", "patrón",
        "indicador", "optimizar", "que paso", "que pasó",
        "historial", "comportamiento"
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
        lambda col: col.str.contains(texto_norm, case=False, na=False, regex=False)
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
# DETECTOR INTELIGENTE DE EQUIPO (más rápido y liviano)
# ==========================================

def detectar_equipo_en_texto(df, texto):
    if df is None or df.empty or not texto:
        return None

    texto = texto.lower()
    vacio = [""] * len(df)

    codigos = df["CODIGO_EXTRAIDO"].astype(str).tolist() if "CODIGO_EXTRAIDO" in df.columns else vacio
    descs = df["DESCRIPCION_EXTRAIDA"].astype(str).tolist() if "DESCRIPCION_EXTRAIDA" in df.columns else vacio

    for codigo, desc in zip(codigos, descs):
        c = codigo.strip().lower()
        d = desc.strip().lower()

        # 🔥 prioridad: codigo
        if c and c in texto:
            return codigo or None

        # 🔥 fallback: descripcion → retorna codigo
        if d and d in texto:
            return codigo or None

    return None


# ==========================================
# CONFIGURACIÓN GEMINI
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("API KEY CARGADA:", GEMINI_API_KEY[:15] if GEMINI_API_KEY else "NO EXISTE")

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "response_mime_type": "text/plain",
}

system_instruction = (
    "Eres un ingeniero senior de mantenimiento experto en gestión de mantenimiento industrial. "
    "Responde siempre en español de manera fluida, natural, amigable y sumamente conversacional (como Gemini). "
    "Tu objetivo es ayudar tanto en consultas de la base de datos como en pautas, inquietudes o dudas del día a día de los colaboradores. "
    "\n\nDIRECTRICES:\n"
    "1. Cuando el usuario consulte sobre registros, órdenes, equipos, técnicos o fallas del historial, "
    "debes apoyarte de forma estricta en la información provista en los contextos del Excel para dar tu respuesta. "
    "Explícala de forma redactada e inteligente con tus propias palabras. No inventes datos específicos si no están presentes.\n"
    "2. Si la información no está en los registros para esa consulta específica, indícalo amablemente diciendo: "
    "'No existe información específica en el registro para esta consulta, pero te puedo brindar recomendaciones técnicas generales sobre el tema.'\n"
    "3. Si el usuario te hace preguntas generales, te saluda, te pide teoría de ingeniería o recomendaciones prácticas para sus tareas diarias, "
    "puedes responder libremente usando todos tus conocimientos técnicos de Ingeniero Senior, sin verte limitado por la base de datos."
)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction
)


# ==========================================
# FASTAPI Y MEMORIA DE SESIONES
# ==========================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

sesiones_chat = {}
memoria_contexto_sheet = {}


@app.get("/")
def inicio():
    # Para probar desde el navegador que el servidor está vivo
    return {"estado": "ok"}


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
# FUNCIÓN AUXILIAR DE CHAT CON MEMORIA
# ==========================================

def obtener_o_crear_chat(session_id):
    if session_id not in sesiones_chat:
        sesiones_chat[session_id] = model.start_chat(history=[])
    return sesiones_chat[session_id]


# ==========================================
# ENDPOINT PRINCIPAL
# ==========================================

@app.post("/chat")
async def chat(
    texto: str = Form(None),
    session_id: str = Form("default_session"),
    archivo: UploadFile = File(None)
):
    print("TEXTO:", texto)

    if archivo:
        print("ARCHIVO RECIBIDO:", archivo.filename)
        print("TIPO:", archivo.content_type)
    else:
        print("NO SE RECIBIÓ ARCHIVO")

    df = None
    texto_extraido = ""
    equipo_detectado = None
    contexto_soporte_interno = ""
    imagen = None

    try:
        # -------- PROCESAMIENTO DE ARCHIVO --------
        if archivo:
            mimetype = archivo.content_type or ""
            bytes_file = await archivo.read()

            print(f"TAMAÑO: {len(bytes_file)/1024/1024:.2f} MB")

            if "pdf" in mimetype:
                texto_extraido = extraer_de_pdf(bytes_file)
            elif "word" in mimetype or "officedocument.wordprocessingml" in mimetype:
                texto_extraido = extraer_de_docx(bytes_file)
            elif "excel" in mimetype or "officedocument.spreadsheetml" in mimetype:
                texto_extraido = extraer_de_excel_adjunto(bytes_file)
            elif "image" in mimetype:
                imagen = Image.open(io.BytesIO(bytes_file))
                imagen = imagen.convert("RGB")
                # Reducir imágenes grandes para ahorrar memoria
                imagen.thumbnail((1600, 1600))
                texto_extraido = "[Imagen enviada por el usuario]"
                print("RESOLUCION:", imagen.size)

            del bytes_file

        if texto_extraido:
            texto = (texto or "") + "\n\nContenido del archivo:\n" + texto_extraido

        texto = texto or ""

        df = obtener_dataframe()
        if df is None or df.empty:
            return {"respuesta": "No se pudo cargar la base de datos.", "tokens_usados": 0}

        equipo_detectado = detectar_equipo_en_texto(df, texto)

        # Clasificadores de consultas
        usar_excel = es_consulta_tecnica(texto)
        es_analitica = es_pregunta_analitica(texto)
        insights = obtener_insights() or {}
        col_equipo = obtener_columna_principal(df)

        if col_equipo is None:
            return {
                "respuesta": "No se encontró columna principal de equipos en la base de datos.",
                "tokens_usados": 0
            }

        texto_lower = texto.lower()

        # ==========================================
        # 🔮 PREDICCIÓN DE RIESGO
        # ==========================================
        if any(p in texto_lower for p in ["riesgo", "fallar", "falla", "probabilidad"]):
            if equipo_detectado and "riesgo_equipos" in insights:
                data = insights["riesgo_equipos"].get(equipo_detectado)
                if data:
                    contexto_soporte_interno += (
                        f"\n[DATOS DE RIESGO DE LA BD]:\n"
                        f"- Equipo: {equipo_detectado}\n"
                        f"- Nivel de riesgo: {data['riesgo']}\n"
                        f"- Score: {data['score']}\n"
                        f"- Motivos: {', '.join(data['motivo'])}\n"
                    )

        # ==========================================
        # 🚨 DETECCIÓN DE ANOMALÍAS
        # ==========================================
        if any(p in texto_lower for p in ["anomalia", "anomalía", "raro", "fuera de lo normal"]):
            anomalias = insights.get("anomalias", {})
            if equipo_detectado:
                data = anomalias.get(equipo_detectado)
                if data:
                    contexto_soporte_interno += (
                        f"\n[ANOMALÍA DETECTADA EN HISTORIAL]:\n"
                        f"- Equipo: {equipo_detectado}\n"
                        f"- Tipo: {data['tipo']} (Nivel: {data['nivel']})\n"
                        f"- Valor actual: {data['valor_actual']} (Promedio: {data['promedio']})\n"
                        f"- Z-score: {data['z_score']}\n"
                    )
            elif anomalias:
                resumen_anomalias = ""
                for eq, data in list(anomalias.items())[:5]:
                    resumen_anomalias += f"- {eq} | Nivel: {data['nivel']} | Z-score: {data['z_score']}\n"
                contexto_soporte_interno += (
                    f"\n[ANOMALÍAS GENERALES EN LA PLANTA (Top 5)]:\n{resumen_anomalias}"
                )

        # ==========================================
        # 📊 HISTORIAL DINÁMICO POR EQUIPO
        # ==========================================
        if equipo_detectado:
            equipo_str = str(equipo_detectado)
            condicion = df[col_equipo].astype(str) == equipo_str

            for c in ["DESCRIPCION_EXTRAIDA", "DESCRIPCIÓN DEL TRABAJO"]:
                if c in df.columns:
                    condicion = condicion | df[c].astype(str).str.contains(
                        equipo_str, case=False, na=False, regex=False
                    )

            df_equipo = df[condicion]

            if not df_equipo.empty:
                memoria_usuario["ultimo_equipo"] = equipo_detectado
                memoria_usuario["ultimo_resultado"] = df_equipo.tail(30).copy()

                lineas_historial = []

                for _, fila in df_equipo.tail(15).iterrows():
                    datos_activos = []

                    for col in df_equipo.columns:
                        if col in ["TEXTO_RAG", "TEXTO_RAG_NORM", "TEXTO_COMPLETO", "CODIGO_NORM", "DESC_NORM"]:
                            continue

                        valor = fila[col]
                        if pd.notna(valor) and str(valor).strip() != "" and str(valor).lower() != "nan":
                            datos_activos.append(f"{col}: {str(valor).strip()}")

                    if datos_activos:
                        lineas_historial.append("• " + " | ".join(datos_activos))

                historial_sintetizado = "\n".join(lineas_historial)

                contexto_soporte_interno += (
                    f"\n[REGISTROS DE MANTENIMIENTO REALES PARA EL EQUIPO {equipo_detectado}]:\n"
                    f"{historial_sintetizado}\n"
                )

        # ==========================================
        # 🔥 MODO ANALÍTICO AVANZADO
        # ==========================================
        if es_analitica and memoria_usuario["ultimo_resultado"] is not None:
            df_eq = memoria_usuario["ultimo_resultado"]
            if "TEXTO_RAG" in df_eq.columns:
                resumen_rag = df_eq["TEXTO_RAG"].astype(str).str.cat(sep=" ")
                contexto_soporte_interno += (
                    f"\n[HISTORIAL ADICIONAL DE ANÁLISIS DE {memoria_usuario['ultimo_equipo']}]:\n"
                    f"{resumen_rag}\n"
                )

        # ==========================================
        # 🌐 BÚSQUEDA EN SHEETS (RAG)
        # ==========================================
        contexto_sheet = ""
        if usar_excel and not contexto_soporte_interno:
            resultado = buscar_en_sheet(texto)
            contexto_sheet = formatear_contexto(resultado)
            if contexto_sheet:
                memoria_contexto_sheet[session_id] = contexto_sheet
        elif session_id in memoria_contexto_sheet and not contexto_soporte_interno:
            contexto_sheet = memoria_contexto_sheet[session_id]

        # ==========================================
        # 💬 CHAT CON GEMINI
        # ==========================================
        chat_sesion = obtener_o_crear_chat(session_id)

        prompt_inyectado = ""
        if contexto_soporte_interno:
            prompt_inyectado += f"\n[DATOS TÉCNICOS HISTÓRICOS Y SENSORIZADOS]:\n{contexto_soporte_interno}\n"
        elif contexto_sheet:
            prompt_inyectado += f"\n[REGISTROS EXCEL DE SOPORTE]:\n{contexto_sheet}\n"

        prompt_final = (
            f"{prompt_inyectado}"
            f"El usuario te hace la siguiente consulta en el chat. "
            f"Responde de forma redactada, natural, amigable y muy fluida como un Ingeniero Senior de Mantenimiento:\n"
            f"Mensaje del Colaborador: {texto}"
        )

        if imagen:
            response = chat_sesion.send_message([prompt_final, imagen])
        else:
            response = chat_sesion.send_message(prompt_final)

        usage = response.usage_metadata
        total_tokens = usage.total_token_count if usage else 0

        print(f"--- Tokens usados (Sesión: {session_id}): {total_tokens} ---")

        return {
            "respuesta": response.text,
            "tokens_usados": total_tokens
        }

    except Exception as e:
        import traceback
        print(f"Error detallado: {traceback.format_exc()}")
        return {"respuesta": f"Lo siento, ocurrió un error interno al procesar tu solicitud: {str(e)}", "tokens_usados": 0}