import io
import re
import gc
import time
import threading
import unicodedata

import requests
import pandas as pd


# ==========================================
# NORMALIZADOR
# ==========================================

def normalizar(texto):
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

    if columnas_usar:
        df["TEXTO_RAG"] = (
            df[columnas_usar]
            .fillna("")
            .astype(str)
            .agg(" | ".join, axis=1)
        )
    else:
        df["TEXTO_RAG"] = ""

    return df


# ==========================================
# CACHE
# ==========================================

cache_excel = {
    "df": None,
    "last_update": 0
}

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/12z2M2H_iE6MAKjgPbDwmt2HaJ7ZQRfx_PL0jDxbQnS8/export?format=csv&gid=955581654"

lock_carga = threading.Lock()

# Columnas que main.py usa directamente: nunca se eliminan
COLUMNAS_PROTEGIDAS = {
    "DESCRIPCIÓN DEL TRABAJO",
    "DESCRIPCION_EXTRAIDA",
    "CODIGO_EXTRAIDO",
    "FECHA (DÍA 01)",
    "FECHA PROGRAMADA",
}


def cache_vigente():
    return cache_excel["df"] is not None and (time.time() - cache_excel["last_update"] <= 3600)


# ==========================================
# CARGA DE DATA (liviana y una sola a la vez)
# ==========================================

def cargar_datos():
    global cache_excel

    if cache_vigente():
        return cache_excel["df"]

    # Solo una carga a la vez (evita dos copias en memoria)
    with lock_carga:
        if cache_vigente():
            return cache_excel["df"]

        try:
            print("🚀 INICIO carga de datos")
            res = requests.get(GOOGLE_SHEET_CSV_URL, timeout=(10, 60))
            res.raise_for_status()
            print("✅ CSV descargado:", round(len(res.content) / 1024 / 1024, 2), "MB")

            # dtype=str: todo como texto (ya no aparecen los ".0")
            df = pd.read_csv(
                io.BytesIO(res.content),
                dtype=str,
                keep_default_na=False,
                encoding="utf-8"
            )
            del res

            df.columns = [str(c).strip() for c in df.columns]
            print("📊 Filas:", len(df), "| Columnas:", len(df.columns))

            # Eliminar columnas totalmente vacías (ahorra mucha memoria)
            vacias = [
                c for c in df.columns
                if c not in COLUMNAS_PROTEGIDAS and (df[c].str.strip() == "").all()
            ]
            df = df.drop(columns=vacias)
            print("📊 Columnas con datos:", len(df.columns))

            col_fecha = "FECHA (DÍA 01)"
            if col_fecha in df.columns:
                df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)

            print("🧠 Construyendo TEXTO_RAG...")
            df = construir_texto_rag(df)

            df["TEXTO_RAG_NORM"] = df["TEXTO_RAG"].apply(normalizar)

            if "CODIGO_EXTRAIDO" in df.columns:
                df["CODIGO_NORM"] = df["CODIGO_EXTRAIDO"].astype(str).apply(normalizar)
            else:
                df["CODIGO_NORM"] = ""

            if "DESCRIPCION_EXTRAIDA" in df.columns:
                df["DESC_NORM"] = df["DESCRIPCION_EXTRAIDA"].astype(str).apply(normalizar)
            else:
                df["DESC_NORM"] = ""

            cache_excel["df"] = df
            cache_excel["last_update"] = time.time()
            gc.collect()

            print("🏁 FIN carga de datos")

        except Exception as e:
            print(f"Error al descargar datos del Sheet: {e}")
            # Si falla, se usa la versión anterior si existe

    return cache_excel["df"]


# ==========================================
# BÚSQUEDA SIMPLE
# ==========================================

def buscar_en_sheet(query):

    df = cargar_datos()

    if df is None or not query:
        return None

    q = normalizar(query)
    if not q:
        return None

    # 🔥 1. Búsqueda por código
    mask_codigo = df["CODIGO_NORM"].str.contains(q, na=False, regex=False)
    if mask_codigo.any():
        return df[mask_codigo].head(5)

    # 🔥 2. Búsqueda por descripción de equipo
    mask_desc = df["DESC_NORM"].str.contains(q, na=False, regex=False)
    if mask_desc.any():
        return df[mask_desc].head(5)

    # 🔥 3. Búsqueda en texto consolidado
    palabras = [p for p in q.split() if len(p) > 3]
    if not palabras:
        return None

    mask_total = pd.Series(False, index=df.index)
    for p in palabras:
        mask_total = mask_total | df["TEXTO_RAG_NORM"].str.contains(p, na=False, regex=False)

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
        return "No se encontró información para ese equipo."

    respuesta = []

    for _, row in df_resultado.iterrows():
        bloque = f"""
🔧 Equipo: {row.get("CODIGO_EXTRAIDO", "")}
📌 Descripción: {row.get("DESCRIPCION_EXTRAIDA", "")}

🛠️ Trabajo realizado:
{row.get("TEXTO_RAG", "")}
"""
        respuesta.append(bloque)

    return "\n\n".join(respuesta)