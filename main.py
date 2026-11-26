import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from collections import OrderedDict
from core.insights import obtener_insights, obtener_columna_principal
import os
import pandas as pd
import io
import re
import threading
import pdfplumber
from docx import Document
from core.rag import buscar_en_sheet, obtener_dataframe, formatear_contexto, normalizar, cargar_datos
from PIL import Image

# Carga el Sheet en segundo plano para que el servidor arranque rápido
threading.Thread(target=cargar_datos, daemon=True).start()


# ==========================================
# AJUSTES DE CONSUMO (puedes modificarlos)
# ==========================================

MAX_REGISTROS_EQUIPO = 10     # registros del equipo que se envían a Gemini
MAX_CHARS_VALOR = 200         # largo máximo de cada dato
MAX_CHARS_NOMBRE_COL = 40     # recorta nombres de columna muy largos
MAX_TURNOS_MEMORIA = 4        # últimas preguntas/respuestas que se recuerdan
MAX_CHARS_RESPUESTA_MEMORIA = 600
MAX_CHARS_ANALITICO = 3000
MAX_SESIONES = 200

COLS_EXCLUIR = {"TEXTO_RAG", "TEXTO_RAG_NORM", "TEXTO_COMPLETO", "CODIGO_NORM", "DESC_NORM"}


# ==========================================
# CLASIFICADORES
# ==========================================

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
        "ocurrió", "cuando", "cuándo", "quien", "quién",
        "ultimo", "último", "veces", "riesgo"
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
        "recomienda", "recomendacion", "recomendación", "evitar", "prevenir", "mejorar",
        "analiza", "analizar", "tendencia", "patron", "patrón",
        "indicador", "optimizar", "que paso", "que pasó",
        "historial", "comportamiento"
    ]

    return any(p in texto for p in palabras)


# ==========================================
# DETECTOR DE EQUIPO
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

        if c and c in texto:
            return codigo or None

        if d and d in texto:
            return codigo or None

    return None


def filtrar_registros_equipo(df, col_equipo, equipo):
    equipo_str = str(equipo)
    condicion = df[col_equipo].astype(str) == equipo_str

    for c in ["DESCRIPCION_EXTRAIDA", "DESCRIPCIÓN DEL TRABAJO"]:
        if c in df.columns:
            condicion = condicion | df[c].astype(str).str.contains(
                equipo_str, case=False, na=False, regex=False
            )

    return df[condicion]


def resumir_registros(df_equipo):
    """Convierte los registros en texto compacto para ahorrar tokens."""
    lineas = []

    for _, fila in df_equipo.tail(MAX_REGISTROS_EQUIPO).iterrows():
        datos = []

        for col in df_equipo.columns:
            if col in COLS_EXCLUIR:
                continue

            valor = fila[col]
            if pd.isna(valor):
                continue

            valor = str(valor).strip()
            if not valor or valor.lower() in ("nan", "nat"):
                continue

            if len(valor) > MAX_CHARS_VALOR:
                valor = valor[:MAX_CHARS_VALOR] + "…"

            nombre = str(col)
            if len(nombre) > MAX_CHARS_NOMBRE_COL:
                nombre = nombre[:MAX_CHARS_NOMBRE_COL] + "…"

            datos.append(f"{nombre}: {valor}")

        if datos:
            lineas.append("• " + " | ".join(datos))

    return "\n".join(lineas)


# ==========================================
# CONFIGURACIÓN GEMINI
# ==========================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("API KEY CARGADA:", GEMINI_API_KEY[:15] if GEMINI_API_KEY else "NO EXISTE")

genai.configure(api_key=GEMINI_API_KEY)

# Puedes cambiar el modelo desde Render (Environment) sin tocar el código
MODELO_GEMINI = os.getenv("MODELO_GEMINI", "gemini-2.5-flash")
print("MODELO GEMINI:", MODELO_GEMINI)

generation_config = {
    "temperature": 0.4,
    "top_p": 0.95,
    "top_k": 40,
    "response_mime_type": "text/plain",
}

system_instruction = (
    "Eres un ingeniero senior de mantenimiento industrial. Respondes en español, de forma clara, natural y amigable. "
    "Ayudas con consultas de la base de datos de mantenimiento y con dudas técnicas del día a día de los colaboradores."
    "\n\nREGLAS:\n"
    "1. Si se te entregan registros de la base de datos, básate estrictamente en ellos. Explícalos con tus palabras; "
    "no inventes datos que no estén presentes.\n"
    "2. Si la información no está en los registros, dilo amablemente: "
    "'No existe información específica en el registro para esta consulta, pero te puedo brindar recomendaciones técnicas generales sobre el tema.'\n"
    "3. Si la conversación ya trata sobre un equipo, asume que las preguntas siguientes se refieren a ese mismo equipo, "
    "salvo que el usuario mencione otro. NO pidas que especifique el equipo si ya hay uno en la conversación.\n"
    "4. Para saludos, teoría o recomendaciones generales puedes responder libremente con tu conocimiento técnico."
    "\n\nEXTENSIÓN Y FORMATO:\n"
    "- Ajusta la extensión a lo que la pregunta realmente necesita: ni más ni menos.\n"
    "- Preguntas puntuales (una fecha, un técnico, un estado): respuesta directa y breve, con el dato y una frase de contexto.\n"
    "- Preguntas de análisis, diagnóstico, historial, comparaciones, procedimientos o paso a paso: responde con toda la "
    "extensión necesaria, bien organizada (puntos o secciones), sin omitir información importante.\n"
    "- Saludos o comentarios simples: 1 a 3 frases.\n"
    "- Empieza siempre con la respuesta directa; luego los detalles que aporten valor.\n"
    "- Sé conciso en la forma, no en el contenido: evita relleno, no repitas la pregunta, no copies los registros "
    "tal cual y no hagas introducciones ni despedidas largas.\n"
    "- Termina siempre tus ideas y frases completas."
)

model = genai.GenerativeModel(
    model_name=MODELO_GEMINI,
    generation_config=generation_config,
    system_instruction=system_instruction
)


# ==========================================
# FASTAPI Y MEMORIA POR SESIÓN
# ==========================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Memoria por usuario: equipo actual, su contexto y las últimas preguntas/respuestas
sesiones = OrderedDict()


def obtener_sesion(session_id):
    if session_id not in sesiones:
        sesiones[session_id] = {
            "equipo": None,
            "contexto_equipo": "",
            "df_equipo": None,
            "contexto_sheet": "",
            "turnos": []
        }
        # Evita que la memoria crezca sin límite
        while len(sesiones) > MAX_SESIONES:
            sesiones.popitem(last=False)
    else:
        sesiones.move_to_end(session_id)

    return sesiones[session_id]


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
# ENDPOINT PRINCIPAL
# ==========================================

@app.post("/chat")
async def chat(
    texto: str = Form(None),
    session_id: str = Form("default_session"),
    archivo: UploadFile = File(None)
):
    print("TEXTO:", texto, "| SESIÓN:", session_id)

    texto_extraido = ""
    imagen = None

    try:
        # -------- PROCESAMIENTO DE ARCHIVO --------
        if archivo:
            mimetype = archivo.content_type or ""
            bytes_file = await archivo.read()
            print(f"ARCHIVO: {archivo.filename} | {mimetype} | {len(bytes_file)/1024/1024:.2f} MB")

            if "pdf" in mimetype:
                texto_extraido = extraer_de_pdf(bytes_file)
            elif "word" in mimetype or "officedocument.wordprocessingml" in mimetype:
                texto_extraido = extraer_de_docx(bytes_file)
            elif "excel" in mimetype or "officedocument.spreadsheetml" in mimetype:
                texto_extraido = extraer_de_excel_adjunto(bytes_file)
            elif "image" in mimetype:
                imagen = Image.open(io.BytesIO(bytes_file)).convert("RGB")
                imagen.thumbnail((1600, 1600))
                texto_extraido = "[Imagen enviada por el usuario]"

            del bytes_file

        texto_usuario = texto or ""
        if texto_extraido:
            # Limita el texto de documentos adjuntos
            texto_usuario += "\n\nContenido del archivo:\n" + texto_extraido[:6000]

        df = obtener_dataframe()
        if df is None or df.empty:
            return {"respuesta": "No se pudo cargar la base de datos.", "tokens_usados": 0}

        col_equipo = obtener_columna_principal(df)
        if col_equipo is None:
            return {
                "respuesta": "No se encontró columna principal de equipos en la base de datos.",
                "tokens_usados": 0
            }

        sesion = obtener_sesion(session_id)
        texto_lower = texto_usuario.lower()

        usar_excel = es_consulta_tecnica(texto_usuario)
        es_analitica = es_pregunta_analitica(texto_usuario)
        pide_riesgo = any(p in texto_lower for p in ["riesgo", "fallar", "falla", "probabilidad"])
        pide_anomalia = any(p in texto_lower for p in ["anomalia", "anomalía", "raro", "fuera de lo normal"])

        # ==========================================
        # 🔎 EQUIPO: nuevo o el de la conversación
        # ==========================================
        equipo_nuevo = detectar_equipo_en_texto(df, texto_usuario)

        if equipo_nuevo and equipo_nuevo != sesion["equipo"]:
            df_equipo = filtrar_registros_equipo(df, col_equipo, equipo_nuevo)
            if not df_equipo.empty:
                sesion["equipo"] = equipo_nuevo
                sesion["df_equipo"] = df_equipo.tail(30).copy()
                sesion["contexto_equipo"] = resumir_registros(df_equipo)
                sesion["contexto_sheet"] = ""
                print("EQUIPO NUEVO:", equipo_nuevo)

        equipo = sesion["equipo"]  # se mantiene en las preguntas siguientes

        # ==========================================
        # 📦 ARMAR CONTEXTO (solo lo necesario)
        # ==========================================
        bloques = []

        necesita_datos = bool(equipo_nuevo) or usar_excel or es_analitica or pide_riesgo or pide_anomalia

        if equipo and sesion["contexto_equipo"]:
            if necesita_datos:
                bloques.append(
                    f"[EQUIPO EN CONVERSACIÓN: {equipo}]\n"
                    f"[ÚLTIMOS REGISTROS DE MANTENIMIENTO]:\n{sesion['contexto_equipo']}"
                )
            else:
                # Pregunta general: solo se recuerda el equipo, sin enviar registros
                bloques.append(f"[EQUIPO EN CONVERSACIÓN: {equipo}]")

        insights = {}
        if pide_riesgo or pide_anomalia:
            insights = obtener_insights() or {}

        # 🔮 Riesgo
        if pide_riesgo and equipo and "riesgo_equipos" in insights:
            data = insights["riesgo_equipos"].get(equipo)
            if data:
                bloques.append(
                    f"[RIESGO DEL EQUIPO {equipo}]: nivel {data['riesgo']}, score {data['score']}, "
                    f"motivos: {', '.join(data['motivo'])}"
                )

        # 🚨 Anomalías
        if pide_anomalia:
            anomalias = insights.get("anomalias", {})
            if equipo and anomalias.get(equipo):
                data = anomalias[equipo]
                bloques.append(
                    f"[ANOMALÍA {equipo}]: tipo {data['tipo']} (nivel {data['nivel']}), "
                    f"valor actual {data['valor_actual']} vs promedio {data['promedio']}, z-score {data['z_score']}"
                )
            elif anomalias:
                resumen = "\n".join(
                    f"- {eq} | Nivel: {d['nivel']} | Z-score: {d['z_score']}"
                    for eq, d in list(anomalias.items())[:5]
                )
                bloques.append(f"[ANOMALÍAS GENERALES (Top 5)]:\n{resumen}")

        # 🔥 Modo analítico
        if es_analitica and sesion["df_equipo"] is not None and "TEXTO_RAG" in sesion["df_equipo"].columns:
            resumen_rag = sesion["df_equipo"]["TEXTO_RAG"].astype(str).str.cat(sep=" ")[:MAX_CHARS_ANALITICO]
            bloques.append(f"[HISTORIAL PARA ANÁLISIS DE {equipo}]:\n{resumen_rag}")

        # 🌐 Búsqueda general en el Sheet (solo si no hay equipo en conversación)
        if not equipo and usar_excel:
            resultado = buscar_en_sheet(texto_usuario)
            if resultado is not None and not resultado.empty:
                sesion["contexto_sheet"] = formatear_contexto(resultado)[:4000]

        if not equipo and sesion["contexto_sheet"] and necesita_datos:
            bloques.append(f"[REGISTROS DE SOPORTE]:\n{sesion['contexto_sheet']}")

        # 💬 Conversación reciente (memoria corta y controlada)
        if sesion["turnos"]:
            historial = "\n".join(
                f"Usuario: {t['usuario']}\nAsistente: {t['asistente']}"
                for t in sesion["turnos"]
            )
            bloques.append(f"[CONVERSACIÓN RECIENTE]:\n{historial}")

        bloques.append(f"[PREGUNTA ACTUAL DEL COLABORADOR]:\n{texto_usuario}")
        prompt_final = "\n\n".join(bloques)

        # ==========================================
        # 🤖 LLAMADA A GEMINI
        # ==========================================
        if imagen:
            response = model.generate_content([prompt_final, imagen])
        else:
            response = model.generate_content(prompt_final)

        try:
            respuesta_texto = response.text
        except Exception:
            respuesta_texto = "No pude generar una respuesta completa. ¿Puedes reformular la pregunta?"

        try:
            print("FINISH:", response.candidates[0].finish_reason)
        except Exception:
            pass

        # Guardar el turno en memoria (recortado)
        sesion["turnos"].append({
            "usuario": (texto or "")[:300],
            "asistente": respuesta_texto[:MAX_CHARS_RESPUESTA_MEMORIA]
        })
        sesion["turnos"] = sesion["turnos"][-MAX_TURNOS_MEMORIA:]

        usage = response.usage_metadata
        total_tokens = getattr(usage, "total_token_count", 0) if usage else 0
        tokens_entrada = getattr(usage, "prompt_token_count", 0) if usage else 0
        tokens_salida = getattr(usage, "candidates_token_count", 0) if usage else 0

        print(f"TOKENS → entrada: {tokens_entrada} | respuesta: {tokens_salida} | total: {total_tokens} | equipo: {equipo}")

        return {
            "respuesta": respuesta_texto,
            "tokens_usados": total_tokens
        }

    except Exception as e:
        import traceback
        print(f"Error detallado: {traceback.format_exc()}")
        return {"respuesta": f"Lo siento, ocurrió un error interno al procesar tu solicitud: {str(e)}", "tokens_usados": 0}