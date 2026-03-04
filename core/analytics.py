import pandas as pd
import re
import unicodedata


# ==========================================
# NORMALIZADOR LOCAL
# ==========================================

def normalizar(texto):
    if not texto:
        return ""
    texto = str(texto).lower()
    texto = "".join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto


# ==========================================
# DETECTOR DE TIPO DE PREGUNTA ANALITICA
# ==========================================

def detectar_tipo_analisis(texto):

    texto = texto.lower()

    if "tecnico" in texto or "técnico" in texto:
        return "tecnico"

    if "falla" in texto:
        return "falla"

    if "equipo" in texto or "linea" in texto or "línea" in texto:
        return "equipo"

    if "mes" in texto or "tendencia" in texto:
        return "tendencia"

    return "general"


# ==========================================
# MOTOR ANALITICO PRINCIPAL
# ==========================================

def ejecutar_analisis(df, texto):

    if df is None or df.empty:
        return None

    tipo = detectar_tipo_analisis(texto)

    resultado = {}

    # ==============================
    # ANALISIS POR TECNICO
    # ==============================
    if tipo == "tecnico":

        col_tecnico = "DIA 1) TEC. N° 01"

        if col_tecnico in df.columns:

            top = (
                df[col_tecnico]
                .value_counts()
                .head(5)
            )

            resultado["tipo"] = "ranking_tecnicos"
            resultado["data"] = top.to_dict()

    # ==============================
    # ANALISIS POR FALLA
    # ==============================
    elif tipo == "falla":

        col_desc = "DESCRIPCIÓN DEL TRABAJO"

        if col_desc in df.columns:

            top = (
                df[col_desc]
                .value_counts()
                .head(5)
            )

            resultado["tipo"] = "ranking_fallas"
            resultado["data"] = top.to_dict()

    # ==============================
    # ANALISIS GENERAL POR EQUIPO
    # ==============================
    elif tipo == "equipo":

        col_desc = "DESCRIPCIÓN DEL TRABAJO"

        if col_desc in df.columns:

            top = (
                df[col_desc]
                .value_counts()
                .head(5)
            )

            resultado["tipo"] = "incidencias_equipo"
            resultado["data"] = top.to_dict()

    # ==============================
    # TENDENCIA MENSUAL
    # ==============================
    elif tipo == "tendencia":

        col_fecha = "FECHA (DÍA 01)"

        if col_fecha in df.columns:

            df_temp = df.copy()
            df_temp[col_fecha] = pd.to_datetime(df_temp[col_fecha], errors='coerce')

            df_temp["mes"] = df_temp[col_fecha].dt.to_period("M")

            tendencia = (
                df_temp["mes"]
                .value_counts()
                .sort_index()
            )

            resultado["tipo"] = "tendencia_mensual"
            resultado["data"] = tendencia.astype(str).to_dict()

    # ==============================
    # GENERAL
    # ==============================
    else:

        total = len(df)

        resultado["tipo"] = "general"
        resultado["data"] = {"total_registros": total}

    return resultado