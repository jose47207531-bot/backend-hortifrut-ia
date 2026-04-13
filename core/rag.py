import pandas as pd
import requests
import io
import time
import re
import unicodedata

from core.insights import guardar_insights

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

# ==========================================
# CACHE
# ==========================================

cache_excel = {
    "df": None,
    "last_update": 0
}

GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1oEVKH1SxHDJtwSx9y3sy1Ui12CqvCWdRTb9bEe_w4D8/export?format=csv&gid=1960130423"

# ==========================================
# CARGA DE DATA
# ==========================================

def cargar_datos():

    global cache_excel

    if cache_excel["df"] is None or (time.time() - cache_excel["last_update"]) > 300:

        res = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)

        df = pd.read_csv(
            io.BytesIO(res.content),
            encoding="utf-8",
            sep=None,
            engine="python"
        ).fillna("")

        # Formatear fecha
        col_fecha = "FECHA (DÍA 01)"
        if col_fecha in df.columns:
            df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')

        df = df.astype(str).replace(r'\.0$', '', regex=True)

        cache_excel["df"] = df
        cache_excel["last_update"] = time.time()

        # 🔥 GENERAR INSIGHTS AUTOMÁTICOS
        guardar_insights(df)

    return cache_excel["df"]

# ==========================================
# BÚSQUEDA SIMPLE (SIN TOP)
# ==========================================

def buscar_en_sheet(query):

    df = cargar_datos()

    if df is None or not query:
        return ""

    query = str(query)

    # 🔢 Búsqueda numérica (orden)
    query_num = re.sub(r'[^0-9]', '', query)

    if len(query_num) >= 4:
        mask = df.astype(str).apply(
            lambda col: col.str.contains(query_num, na=False)
        )
        resultado = df[mask.any(axis=1)]

        if not resultado.empty:
            return resultado.head(20).to_markdown(index=False)

    # 🔍 Búsqueda por texto (en TODAS las columnas)
    query_norm = normalizar(query)

    mask = df.astype(str).apply(
        lambda col: col.str.contains(query_norm, case=False, na=False)
    )

    resultado = df[mask.any(axis=1)]

    if resultado.empty:
        return ""

    return resultado.head(20).to_markdown(index=False)

# ==========================================
# ACCESO GLOBAL
# ==========================================

def obtener_dataframe():
    return cargar_datos()