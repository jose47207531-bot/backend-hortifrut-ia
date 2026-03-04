import pandas as pd
import numpy as np
import requests
import io
import time
import re
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = [
    "que", "hubo", "en", "la", "el", "los", "las",
    "de", "del", "al", "y", "o", "un", "una",
    "hay", "existe", "trabajo", "trabajos",
    "línea", "linea", "por", "para", "con"
]

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

def limpiar_query(query):
    palabras = query.lower().split()
    palabras_filtradas = [p for p in palabras if p not in STOPWORDS]
    return normalizar(" ".join(palabras_filtradas))
# ==========================================
# BÚSQUEDA INTELIGENTE EN GOOGLE SHEET
# ==========================================
cache_excel = {
    "df": None,
    "last_update": 0,
    "vectorizer": None,
    "tfidf_matrix": None
}



GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1oEVKH1SxHDJtwSx9y3sy1Ui12CqvCWdRTb9bEe_w4D8/export?format=csv&gid=1960130423"
def buscar_en_sheet(query):
    global cache_excel

    try:
        # ==========================================
        # 1️⃣ ACTUALIZAR CACHE CADA 5 MINUTOS
        # ==========================================
        
        if cache_excel["df"] is None or (time.time() - cache_excel["last_update"]) > 300:

            res = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)

            df_raw = pd.read_csv(
                io.BytesIO(res.content),
                encoding="utf-8",
                sep=None,
                engine="python"
            ).fillna("")

            # Formatear fecha si existe
            col_fecha = "FECHA (DÍA 01)"
            if col_fecha in df_raw.columns:
                df_raw[col_fecha] = pd.to_datetime(
                    df_raw[col_fecha], errors='coerce'
                ).dt.strftime('%d-%m-%Y')

            df_raw = df_raw.astype(str).replace(r'\.0$', '', regex=True)

            # ==========================================
            # 2️⃣ CREAR COLUMNA DE BÚSQUEDA UNIFICADA
            # ==========================================
            df_raw["busqueda"] = (
                df_raw.astype(str)
                .agg(' '.join, axis=1)
                .apply(normalizar)
            )

            # ==========================================
            # 3️⃣ CREAR MODELO TF-IDF
            # ==========================================
            vectorizer = TfidfVectorizer(
            ngram_range=(1,2),   # detecta frases como "linea belando"
            min_df=1,
            sublinear_tf=True)
            tfidf_matrix = vectorizer.fit_transform(df_raw["busqueda"])

            # Guardar en cache
            cache_excel["df"] = df_raw
            cache_excel["vectorizer"] = vectorizer
            cache_excel["tfidf_matrix"] = tfidf_matrix
            cache_excel["last_update"] = time.time()

        # ==========================================
        # 4️⃣ RECUPERAR DEL CACHE
        # ==========================================
        df = cache_excel["df"]
        vectorizer = cache_excel["vectorizer"]
        tfidf_matrix = cache_excel["tfidf_matrix"]

                # ==========================================
        # 🔢 DETECTAR BÚSQUEDA NUMÉRICA (ORDEN)
        # ==========================================

        query_solo_num = re.sub(r'[^0-9]', '', str(query))

        if len(query_solo_num) >= 4:  # mínimo 4 dígitos para evitar ruido

            mask_numerica = df.astype(str).apply(
                lambda col: col.str.contains(query_solo_num, na=False)
            )

            coincidencias = df[mask_numerica.any(axis=1)]

            if not coincidencias.empty:

                columnas_importantes = [
                    "N° DE ORDEN",
                    "FECHA (DÍA 01)",
                    "DESCRIPCIÓN DEL TRABAJO",
                    "STATUS 1",
                    "DIA 1) TEC. N° 01"
                ]

                columnas_validas = [
                    c for c in columnas_importantes if c in coincidencias.columns
                ]

                return coincidencias[columnas_validas].head(15).to_markdown(index=False)

        # ==========================================
        # 5️⃣ LIMPIAR CONSULTA
        # ==========================================
        query_limpia = limpiar_query(query)
        

        if not query_limpia.strip():
            return ""

        query_normalizada = normalizar(query_limpia)

        

        # 6️⃣ TRANSFORMAR QUERY A VECTOR
        # ==========================================
        query_vec = vectorizer.transform([query_normalizada])

        similitudes = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # ==========================================
        # 7️⃣ OBTENER TOP 15 MÁS SIMILARES
        # ==========================================
        top_indices = similitudes.argsort()[-15:][::-1]

        resultado = df.iloc[top_indices]

        if resultado.empty:
         return ""
        # Si existe columna de fecha, ordenar por fecha descendente
        col_fecha = "FECHA (DÍA 01)"
        if col_fecha in resultado.columns:
         resultado[col_fecha] = pd.to_datetime(resultado[col_fecha], errors='coerce')
         resultado = resultado.sort_values(by=col_fecha, ascending=False)

        # Tomar solo los últimos 15
        resultado = resultado.head(15)            

        columnas_importantes = [
            "N° DE ORDEN",
            "FECHA (DÍA 01)",
            "DESCRIPCIÓN DEL TRABAJO",
            "STATUS 1",
            "DIA 1) TEC. N° 01"
        ]

        columnas_validas = [
            c for c in columnas_importantes if c in resultado.columns
        ]

        return resultado[columnas_validas].head(15).to_markdown(index=False)
    
    
    except Exception as e:
        print("Error búsqueda TF-IDF:", e)
        return ""
def obtener_dataframe():
    return cache_excel.get("df")  