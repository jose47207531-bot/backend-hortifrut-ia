import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# ==========================================
# 🔥 FUNCIÓN NUEVA (CLAVE)
# ==========================================
def obtener_columna_principal(df):
    if "CODIGO_EXTRAIDO" in df.columns:
        return "CODIGO_EXTRAIDO"
    if "DESCRIPCION_EXTRAIDA" in df.columns:
        return "DESCRIPCION_EXTRAIDA"
    return None


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
# 🔥 DETECCIÓN DE EQUIPO MEJORADA
# ==========================================

def detectar_equipo_desde_texto(df, texto):

    texto = normalizar(texto)

    col_principal = obtener_columna_principal(df)

    if col_principal is None:
        return None

    # 🔹 1. Búsqueda directa por código
    coincidencias = df[
        df[col_principal].astype(str).str.contains(texto, na=False)
    ]

    if not coincidencias.empty:
        return coincidencias[col_principal].iloc[0]

    # 🔹 2. Búsqueda por descripción
    if "DESCRIPCION_EXTRAIDA" in df.columns:

        df_temp = df.copy()
        df_temp["desc_norm"] = df_temp["DESCRIPCION_EXTRAIDA"].apply(normalizar)

        match = df_temp[
            df_temp["desc_norm"].str.contains(texto, na=False)
        ]

        if not match.empty:
            return match[col_principal].iloc[0]

    # 🔹 3. Búsqueda en TEXTO COMPLETO (🔥 NUEVO)
    col_texto = "TEXTO_COMPLETO" if "TEXTO_COMPLETO" in df.columns else "DESCRIPCIÓN DEL TRABAJO"

    if col_texto in df.columns:

        df_temp = df.copy()
        df_temp["texto_norm"] = df_temp[col_texto].apply(normalizar)

        match = df_temp[
            df_temp["texto_norm"].str.contains(texto, na=False)
        ]

        if not match.empty:
            return match[col_principal].iloc[0]

    return None


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
# 🔥 ANALISIS TECNICO MEJORADO
# ==========================================

def generar_analisis_tecnico_avanzado(df, consulta):

    if df is None:
        return "No hay datos históricos cargados."

    consulta = normalizar(consulta)

    filtro = df.astype(str).apply(
        lambda col: col.str.contains(consulta, case=False, na=False)
    )

    df_filtrado = df[filtro.any(axis=1)]

    if df_filtrado.empty:
        return "No se encontraron eventos históricos similares."

    col_texto = "TEXTO_COMPLETO" if "TEXTO_COMPLETO" in df_filtrado.columns else "DESCRIPCIÓN DEL TRABAJO"

    conteo = df_filtrado[col_texto].value_counts()

    respuesta = "Basado en historial, las intervenciones más frecuentes son:\n\n"

    for desc, cantidad in conteo.items():
        respuesta += f"- {desc} ({cantidad} veces)\n"

    return respuesta


# ==========================================
# MOTOR ANALITICO PRINCIPAL
# ==========================================

def ejecutar_analisis(df, texto):

    if df is None or df.empty:
        return None

    tipo = detectar_tipo_analisis(texto)

    resultado = {}

    col_principal = obtener_columna_principal(df)

    # ==============================
    # ANALISIS POR TECNICO
    # ==============================
    if tipo == "tecnico":

        col_tecnico = "DIA 1) TEC. N° 01"

        if col_tecnico in df.columns:

            top = df[col_tecnico].value_counts()

            resultado["tipo"] = "ranking_tecnicos"
            resultado["data"] = top.to_dict()

    # ==============================
    # ANALISIS POR FALLA
    # ==============================
    elif tipo == "falla":

        col_texto = "TEXTO_COMPLETO" if "TEXTO_COMPLETO" in df.columns else "DESCRIPCIÓN DEL TRABAJO"

        if col_texto in df.columns:

            top = df[col_texto].value_counts()

            resultado["tipo"] = "ranking_fallas"
            resultado["data"] = top.to_dict()

    # ==============================
    # ANALISIS GENERAL POR EQUIPO
    # ==============================
    elif tipo == "equipo":

        if col_principal is not None:

            equipo_detectado = detectar_equipo_desde_texto(df, texto)

            if equipo_detectado:
                df_filtrado = df[df[col_principal] == equipo_detectado].copy()
            else:
                df_filtrado = df.copy()

            if "FECHA (DÍA 01)" in df_filtrado.columns:
                df_filtrado["FECHA (DÍA 01)"] = pd.to_datetime(
                    df_filtrado["FECHA (DÍA 01)"], errors='coerce'
                )

            col_texto = "TEXTO_COMPLETO" if "TEXTO_COMPLETO" in df_filtrado.columns else "DESCRIPCIÓN DEL TRABAJO"

            conteo = df_filtrado[col_texto].value_counts()

            resultado["tipo"] = "incidencias_equipo"
            resultado["equipo"] = equipo_detectado
            resultado["total"] = len(df_filtrado)
            resultado["tipos_trabajo"] = len(conteo)
            resultado["data"] = conteo.to_dict()

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