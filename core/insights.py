import pandas as pd
import unicodedata
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 🔥 Memoria global
INSIGHTS = {}

# ==========================================
# NORMALIZADOR
# ==========================================
def obtener_columna_principal(df):
    if "CODIGO_EXTRAIDO" in df.columns:
        return "CODIGO_EXTRAIDO"
    if "DESCRIPCION_EXTRAIDA" in df.columns:
        return "DESCRIPCION_EXTRAIDA"
    return None

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

def construir_texto_completo(df):

    columnas_base = []

    if "DESCRIPCIÓN DEL TRABAJO" in df.columns:
        columnas_base.append("DESCRIPCIÓN DEL TRABAJO")

    columnas_tareas = [col for col in df.columns if "TAREA" in col.upper()]
    columnas_resp = [col for col in df.columns if "RESPONSABLE" in col.upper()]

    columnas_usar = columnas_base + columnas_tareas + columnas_resp

    if not columnas_usar:
        return df

    df = df.copy()

    df["TEXTO_COMPLETO"] = (
        df[columnas_usar]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .apply(normalizar)
    )

    return df


def detectar_anomalias(df):

    if df is None or df.empty:
        return {}

    col_principal = obtener_columna_principal(df)

    if "FECHA (DÍA 01)" not in df.columns or col_principal is None:
        return {}

    df_temp = df.copy()

    # Asegurar fecha

    df_temp["FECHA (DÍA 01)"] = pd.to_datetime(
        df_temp["FECHA (DÍA 01)"], errors="coerce"
    )

    df_temp["mes"] = df_temp["FECHA (DÍA 01)"].dt.to_period("M")

    tabla = (
        df_temp.groupby([col_principal, "mes"])
        .size()
        .unstack(fill_value=0)
    )

    anomalias = {}

    for equipo in tabla.index:

        serie = tabla.loc[equipo]

        if len(serie) < 3:
            continue

        promedio = serie.mean()
        std = serie.std()
        ultimo = serie.iloc[-1]

        # 🔥 Regla de anomalía

        if std == 0:
            continue

        z_score = (ultimo - promedio) / std

        if z_score > 2:

            anomalias[equipo] = {
                "tipo": "incremento_anormal",
                "valor_actual": int(ultimo),
                "promedio": round(promedio, 2),
                "desviacion": round(std, 2),
                "z_score": round(z_score, 2),
                "nivel": "ALTO" if z_score > 3 else "MEDIO"
            }

    return anomalias


# ==========================================
# 🔥 CLUSTERING INTELIGENTE
# ==========================================
def generar_clusters(df, n_clusters=8):

    if "DESCRIPCIÓN DEL TRABAJO" not in df.columns:
        return df, {}

    col_texto = "TEXTO_COMPLETO" if "TEXTO_COMPLETO" in df.columns else "DESCRIPCIÓN DEL TRABAJO"

    textos = df[col_texto].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        stop_words=None,
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(textos)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["CLUSTER"] = kmeans.fit_predict(X)

    # 🔥 Crear resumen de clusters

    resumen_clusters = {}

    for cluster_id in sorted(df["CLUSTER"].unique()):
        subset = df[df["CLUSTER"] == cluster_id]

        col_texto = "TEXTO_COMPLETO" if "TEXTO_COMPLETO" in subset.columns else "DESCRIPCIÓN DEL TRABAJO"

        ejemplos = subset[col_texto].head(3).tolist()

        resumen_clusters[cluster_id] = {
            "total": len(subset),
            "ejemplos": ejemplos
        }

    return df, resumen_clusters


def calcular_riesgo_equipos(df):

    if df is None or df.empty:
        return {}

    col_principal = obtener_columna_principal(df)

    if col_principal is None:
        return {}

    df = df.copy()

     # 🔹 Fecha

    if "FECHA (DÍA 01)" in df.columns:
        df["FECHA (DÍA 01)"] = pd.to_datetime(
            df["FECHA (DÍA 01)"], errors="coerce"
        )

    riesgo_equipos = {}

    for eq in df[col_principal].dropna().unique():

        df_eq = df[df[col_principal] == eq].copy()

        total = len(df_eq)

        frecuencia_score = min(total * 2, 40)

        correctivo_score = 0

        if "TIPO DE MANTENIMIENTO" in df_eq.columns:

            tipos = (
                df_eq["TIPO DE MANTENIMIENTO"]
                .astype(str)
                .str.upper()
                .value_counts(normalize=True)
            )

            correctivo_pct = tipos.get("CORRECTIVO", 0)

            correctivo_score = correctivo_pct * 40

        tendencia_score = 0

        if "FECHA (DÍA 01)" in df_eq.columns:

            df_eq["mes"] = df_eq["FECHA (DÍA 01)"].dt.to_period("M")

            if not df_eq["mes"].isna().all():

                max_mes = df_eq["mes"].max()

                ultimos = df_eq[df_eq["mes"] >= (max_mes - 2)]
                anteriores = df_eq[df_eq["mes"] < (max_mes - 2)]

                if len(anteriores) > 0:
                    ratio = len(ultimos) / len(anteriores)

                    if ratio > 2:
                        tendencia_score = 20
                    elif ratio > 1.3:
                        tendencia_score = 10

        variedad_score = 0

        if "DESCRIPCIÓN DEL TRABAJO" in df_eq.columns:

            col_texto = "TEXTO_COMPLETO" if "TEXTO_COMPLETO" in df_eq.columns else "DESCRIPCIÓN DEL TRABAJO"

            tipos_trabajo = df_eq[col_texto].value_counts()
            variedad_real = len(tipos_trabajo[tipos_trabajo > 1])

            if variedad_real > 10:
                variedad_score = 10
            elif variedad_real > 5:
                variedad_score = 5

        score = (
            frecuencia_score +
            correctivo_score +
            tendencia_score +
            variedad_score
        )

        if score >= 75:
            nivel = "ALTO"
        elif score >= 45:
            nivel = "MEDIO"
        else:
            nivel = "BAJO"

        motivos = []

        if frecuencia_score > 25:
            motivos.append("Alta frecuencia de intervenciones")

        if correctivo_score > 20:
            motivos.append("Alto porcentaje de mantenimiento correctivo")

        if tendencia_score >= 10:
            motivos.append("Incremento reciente de eventos")

        if variedad_score > 0:
            motivos.append("Alta diversidad de fallas detectadas")

        riesgo_equipos[eq] = {
            "riesgo": nivel,
            "score": round(score, 2),
            "motivo": motivos,
            "total_eventos": total
        }

    return riesgo_equipos


# ==========================================
# 🔥 INSIGHTS PRINCIPAL
# ==========================================
def generar_insights(df):

    if df is None or df.empty:
        return {}

    insights = {}

    col_principal = obtener_columna_principal(df)

    if col_principal:
        equipo_stats = (
            df.groupby(col_principal)
            .size()
            .sort_values(ascending=False)
            .head(20)
        )

        insights["fallas_por_equipo"] = equipo_stats.to_dict()

    df = construir_texto_completo(df)
   
    if "CLUSTER" in df.columns:

        cluster_stats = (
            df.groupby("CLUSTER")
            .size()
            .sort_values(ascending=False)
        )

        insights["fallas_por_cluster"] = cluster_stats.to_dict()

    if col_principal and "DESCRIPCIÓN DEL TRABAJO" in df.columns:

        historial = {}

        for eq in df[col_principal].dropna().unique():

            df_eq = df[df[col_principal] == eq]

            col_texto = "TEXTO_COMPLETO" if "TEXTO_COMPLETO" in df_eq.columns else "DESCRIPCIÓN DEL TRABAJO"

            historial[eq] = {
                "total": len(df_eq),
                "trabajos": df_eq[col_texto].value_counts().to_dict()
            }

        insights["historial_equipos"] = historial

    if "FECHA (DÍA 01)" in df.columns:

        df_temp = df.copy()
        df_temp["FECHA (DÍA 01)"] = pd.to_datetime(df_temp["FECHA (DÍA 01)"], errors="coerce")

        df_temp["mes"] = df_temp["FECHA (DÍA 01)"].dt.to_period("M")

        tendencia = df_temp["mes"].value_counts().sort_index()

        insights["tendencia"] = tendencia.astype(str).to_dict()

    if "TIPO DE MANTENIMIENTO" in df.columns:

        tipo_dist = df["TIPO DE MANTENIMIENTO"].value_counts()
        insights["tipos_mantenimiento"] = tipo_dist.to_dict()

    if col_principal and "TIPO DE MANTENIMIENTO" in df.columns:

        tipo_por_equipo = {}

        for eq in df[col_principal].dropna().unique():

            df_eq = df[df[col_principal] == eq]

            conteo = df_eq["TIPO DE MANTENIMIENTO"].value_counts()

            tipo_por_equipo[eq] = conteo.to_dict()

        insights["tipo_mantenimiento_por_equipo"] = tipo_por_equipo

    if "FECHA (DÍA 01)" in df.columns and "TIPO DE MANTENIMIENTO" in df.columns:

        df_temp = df.copy()

        df_temp["FECHA (DÍA 01)"] = pd.to_datetime(
            df_temp["FECHA (DÍA 01)"], errors="coerce"
        )

        df_temp["mes"] = df_temp["FECHA (DÍA 01)"].dt.to_period("M")

        tendencia_tipo = (
            df_temp.groupby(["mes", "TIPO DE MANTENIMIENTO"])
            .size()
            .unstack(fill_value=0)
        )

        insights["tendencia_tipo_mantenimiento"] = tendencia_tipo.astype(str).to_dict()

    riesgo = calcular_riesgo_equipos(df.copy())
    insights["riesgo_equipos"] = riesgo

    anomalias = detectar_anomalias(df)
    insights["anomalias"] = anomalias

    return insights


# ==========================================
# GUARDAR EN MEMORIA
# ==========================================
def guardar_insights(df):
    global INSIGHTS
    INSIGHTS = generar_insights(df)


# ==========================================
# OBTENER
# ==========================================
def obtener_insights():
    return INSIGHTS