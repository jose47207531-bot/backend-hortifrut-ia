import pandas as pd
import requests
import io
import time
import re


from core.insights import guardar_insights

# ==========================================
# NORMALIZADOR
# ==========================================

def normalizar(texto):
    import unicodedata, re
    if not texto:
        return ""
    texto = str(texto).lower()
    texto = "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return texto.strip()

def construir_texto_rag(df):

    columnas_base = [
        "DESCRIPCIÓN DEL TRABAJO",
        "FECHA PROGRAMADA",
        "Descripción del Trabajo Realizado Indique lo realizado Valores y/o resultados de pruebas realizadas si es necesario puede hacer algún esquema en el reverso use hojas en blanco para notificar si es necesario engrampandola adecuadamente.",
        "Observaciones y/o Recomendaciones Pendientes de Realizar Generar el AVISO correspondiente."
    ]

    columnas_tareas = [col for col in df.columns if "TAREA" in col.upper()]
    columnas_resp = [col for col in df.columns if "RESPONSABLE" in col.upper()]

    columnas_usar = columnas_base + columnas_tareas + columnas_resp

    columnas_usar = [c for c in columnas_usar if c in df.columns]

    df = df.copy()

    df["TEXTO_RAG"] = (
        df[columnas_usar]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
    )

    return df



# ==========================================
# CACHE
# ==========================================

cache_excel = {
    "df": None,
    "last_update": 0
}

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/12z2M2H_iE6MAKjgPbDwmt2HaJ7ZQRfx_PL0jDxbQnS8/export?format=csv&gid=955581654"

# ==========================================
# CARGA DE DATA
# ==========================================

def cargar_datos():

    global cache_excel

    if cache_excel["df"] is None or (time.time() - cache_excel["last_update"] > 3600):
        
        try:
            res = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)
            res.raise_for_status()

            df = pd.read_csv(
             io.BytesIO(res.content),
             encoding="utf-8",
             sep=None,
             engine="python"
             ).fillna("")

            # Formatear fecha
            col_fecha = "FECHA (DÍA 01)"
            if col_fecha in df.columns:
              df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce', dayfirst=True)

            df = df.astype(str).replace(r'\.0$', '', regex=True)

            # 🔥 NUEVO: construir texto RAG
            df = construir_texto_rag(df)

            # 🔥 NORMALIZACIONES (UNA SOLA VEZ)
            df["TEXTO_RAG_NORM"] = df["TEXTO_RAG"].apply(normalizar)
            df["CODIGO_NORM"] = df["CODIGO_EXTRAIDO"].astype(str).apply(normalizar)
            df["DESC_NORM"] = df["DESCRIPCION_EXTRAIDA"].astype(str).apply(normalizar)

            cache_excel["df"] = df
            cache_excel["last_update"] = time.time()
            # 🔥 GENERAR INSIGHTS AUTOMÁTICOS
            guardar_insights(df)

        except Exception as e:
            print(f"Error al descargar datos del Sheet: {e}")
            return None        
    return cache_excel["df"]

# ==========================================
# BÚSQUEDA SIMPLE (SIN TOP)
# ==========================================

def buscar_en_sheet(query):

    df = cargar_datos()

    if df is None or not query:
        return None

    q = normalizar(query)

    # 🔥 1. Búsqueda por código (rápida)
    mask_codigo = df["CODIGO_NORM"].str.contains(q, na=False)

    if mask_codigo.any():
        return df[mask_codigo].head(5)

    # 🔥 2. Búsqueda por descripción de equipo
    mask_desc = df["DESC_NORM"].str.contains(q, na=False)

    if mask_desc.any():
        return df[mask_desc].head(5)

    # 🔥 3. Búsqueda en texto consolidado
    palabras = [p for p in q.split() if len(p) > 3]

    if not palabras:
        return None

    mask_total = False

    for p in palabras:
        mask_total = mask_total | df["TEXTO_RAG_NORM"].str.contains(p, na=False)

    resultado = df[mask_total]

    if resultado.empty:
        return None

    return resultado.head(5)

# ==========================================
# ACCESO GLOBAL
# ==========================================

def obtener_dataframe():
    return cargar_datos()

def formatear_contexto(df_resultado):

    if df_resultado is None or df_resultado.empty:
        return ""

    cols = [
        "CODIGO_EXTRAIDO",
        "DESCRIPCION_EXTRAIDA",
        "TEXTO_RAG"
    ]

    df_small = df_resultado[cols].copy()

    return df_small.to_dict(orient="records")