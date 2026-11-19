"""
Asistente MercaAndes — búsqueda híbrida (Semántica + BM25 + RRF) + re-ranking.
Ejecutar:  streamlit run app.py
"""
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

st.set_page_config(page_title="Asistente MercaAndes", page_icon="🛒", layout="wide")

COLECCION = "mercaandes"
MODELO_CHAT = "gemini-3.5-flash-lite"
MODELO_EMBEDDING = "gemini-embedding-001"
DIMENSIONES = 768
MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


# ------------------------------------------------------------------
# 1. Conexiones y recursos pesados (se crean UNA sola vez)
# ------------------------------------------------------------------
def secreto(nombre):
    # Primero variables de entorno (Docker / EasyPanel), luego .streamlit/secrets.toml
    if nombre in os.environ:
        return os.environ[nombre]
    return st.secrets[nombre]


@st.cache_resource
def conectar():
    gemini = genai.Client(api_key=secreto("GEMINI_API_KEY"))
    qdrant = QdrantClient(url=secreto("QDRANT_URL"), api_key=secreto("QDRANT_API_KEY"))
    return gemini, qdrant


@st.cache_resource(show_spinner="Descargando el modelo de re-ranking (solo la primera vez)...")
def cargar_reranker():
    return CrossEncoder(MODELO_RERANKER)


gemini, qdrant = conectar()


# ------------------------------------------------------------------
# 2. Funciones del notebook (misma lógica)
# ------------------------------------------------------------------
def cargar_fragmentos():
    todos = []
    offset = None

    while True:
        puntos, offset = qdrant.scroll(
            collection_name=COLECCION,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        todos.extend(puntos)

        if offset is None:
            break

    registros = []
    for punto in todos:
        p = punto.payload
        registros.append({
            "id": p.get("id", str(punto.id)),
            "documento": p.get("titulo", p.get("doc_id", "Documento")),
            "texto": p.get("texto", "")
        })

    return pd.DataFrame(registros)


def tokenizar(texto):
    return re.findall(r"[a-záéíóúñ0-9]+", texto.lower())


@st.cache_resource(show_spinner="Cargando fragmentos de Qdrant y creando el índice BM25...")
def preparar_indice():
    documentos = cargar_fragmentos()
    bm25 = BM25Okapi([tokenizar(texto) for texto in documentos["texto"]])
    return documentos, bm25


documentos, bm25 = preparar_indice()
reranker = cargar_reranker()


def buscar_bm25(pregunta, k=5):
    scores = bm25.get_scores(tokenizar(pregunta))
    posiciones = np.argsort(scores)[::-1][:k]

    resultados = documentos.iloc[posiciones].copy()
    resultados["score_bm25"] = scores[posiciones]
    return resultados[resultados["score_bm25"] > 0]


def crear_embedding(texto):
    respuesta = gemini.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMENSIONES
        )
    )
    vector = np.array(respuesta.embeddings[0].values, dtype=np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


def buscar_semantica(pregunta, k=5):
    resultado = qdrant.query_points(
        collection_name=COLECCION,
        query=crear_embedding(pregunta),
        limit=k,
        with_payload=True
    )

    filas = []
    for punto in resultado.points:
        p = punto.payload
        filas.append({
            "id": p.get("id", str(punto.id)),
            "documento": p.get("titulo", p.get("doc_id", "Documento")),
            "texto": p.get("texto", ""),
            "score_semantico": punto.score
        })

    return pd.DataFrame(filas)


def buscar_hibrido(pregunta, k=5, candidatos=10, constante=60):
    semantica = buscar_semantica(pregunta, k=candidatos)
    palabras = buscar_bm25(pregunta, k=candidatos)

    puntajes = {}

    for rango, fila in enumerate(semantica.itertuples(), start=1):
        puntajes[fila.id] = puntajes.get(fila.id, 0) + 1 / (constante + rango)

    for rango, fila in enumerate(palabras.itertuples(), start=1):
        puntajes[fila.id] = puntajes.get(fila.id, 0) + 1 / (constante + rango)

    ids_ordenados = sorted(puntajes, key=puntajes.get, reverse=True)[:k]

    base = documentos.set_index("id")
    salida = base.loc[ids_ordenados].reset_index()
    salida["score_rrf"] = [puntajes[i] for i in ids_ordenados]

    return salida


def rerankear(pregunta, fragmentos, k=5):
    pares = [(pregunta, texto) for texto in fragmentos["texto"]]
    scores = reranker.predict(pares)

    salida = fragmentos.copy()
    salida["score_rerank"] = scores
    salida = salida.sort_values("score_rerank", ascending=False).head(k)
    return salida.reset_index(drop=True)


def buscar_con_rerank(pregunta, k=5, candidatos=20):
    hibrido = buscar_hibrido(pregunta, k=candidatos, candidatos=candidatos)
    hibrido["posicion_rrf"] = range(1, len(hibrido) + 1)
    return rerankear(pregunta, hibrido, k=k)


SYSTEM_PROMPT = """
Eres el asistente de atención al cliente de MercaAndes.
Responde en español y utiliza solamente el contexto entregado.
No inventes información. Si el contexto no responde la pregunta,
di que no tienes información suficiente.
Cita los identificadores de los fragmentos utilizados.
"""


def responder(pregunta, buscador, k=5):
    # Igual que responder_hibrido / responder_con_rerank, pero con el buscador elegido en la barra lateral
    fragmentos = buscador(pregunta, k=k)

    contexto = "\n\n".join(
        f"[{fila.id}] {fila.documento}\n{fila.texto}"
        for fila in fragmentos.itertuples()
    )

    respuesta = gemini.models.generate_content(
        model=MODELO_CHAT,
        contents=f"CONTEXTO:\n{contexto}\n\nPREGUNTA:\n{pregunta}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2
        )
    )

    return respuesta.text, fragmentos


METODOS = {
    "RRF + Re-ranker": buscar_con_rerank,
    "RRF": buscar_hibrido,
    "Semántica": buscar_semantica,
    "BM25": buscar_bm25,
}

GOLDEN = [
    {"pregunta": "Ya no quiero un producto que me llegó, ¿cuántos días tengo para devolverlo?",
     "relevantes": ["D01-04", "D05-03"]},
    {"pregunta": "¿Cuánto cuesta el envío a Cusco y en cuántos días llega?",
     "relevantes": ["D03-02"]},
    {"pregunta": "Pagué con Yape, ¿en cuánto tiempo me devuelven el dinero?",
     "relevantes": ["D01-09", "D05-04"]},
    {"pregunta": "¿Cuánto dura la garantía de una laptop AndesTech?",
     "relevantes": ["D09-05"]},
    {"pregunta": "Soy cliente nivel Oro y vivo en Arequipa, ¿tengo que pagar el envío si devuelvo algo que ya no quiero?",
     "relevantes": ["D01-07", "D07-06"]},
]


def posicion_primer_relevante(ids, relevantes):
    for posicion, i in enumerate(ids, start=1):
        if i in relevantes:
            return posicion
    return None


def columna_score(df):
    # Nombre de la columna de score que trae cada buscador
    return [c for c in df.columns if c.startswith("score_")][-1]   # el último = el que decide el orden


# ------------------------------------------------------------------
# 3. Interfaz
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    metodo = st.selectbox("Método de búsqueda", list(METODOS))
    k = st.slider("Fragmentos enviados a Gemini (k)", 3, 10, 5)
    st.caption(f"{len(documentos)} fragmentos cargados de `{COLECCION}`")
    if st.button("🔄 Recargar fragmentos"):
        preparar_indice.clear()
        st.rerun()

st.title("🛒 Asistente MercaAndes")
tab_chat, tab_comparar, tab_eval = st.tabs(["💬 Asistente", "🔎 Comparar búsquedas", "📊 Evaluación"])

# --- Tab 1: chat ---------------------------------------------------
with tab_chat:
    if "historial" not in st.session_state:
        st.session_state.historial = []

    for mensaje in st.session_state.historial:
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["texto"])
            if "fuentes" in mensaje:
                with st.expander("Fuentes"):
                    st.dataframe(mensaje["fuentes"], hide_index=True)

    if pregunta := st.chat_input("Escribe tu pregunta..."):
        st.session_state.historial.append({"rol": "user", "texto": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner(f"Buscando con {metodo}..."):
                respuesta, fuentes = responder(pregunta, METODOS[metodo], k=k)
            st.markdown(respuesta)
            tabla_fuentes = fuentes[["id", "documento", columna_score(fuentes)]]
            with st.expander("Fuentes"):
                st.dataframe(tabla_fuentes, hide_index=True)

        st.session_state.historial.append(
            {"rol": "assistant", "texto": respuesta, "fuentes": tabla_fuentes})

# --- Tab 2: comparar los 4 métodos ---------------------------------
with tab_comparar:
    pregunta_cmp = st.text_input("Pregunta", "¿Cuánto cuesta enviar un pedido a Cusco?")
    if st.button("Comparar", type="primary"):
        columnas = st.columns(len(METODOS))
        for col, (nombre, buscador) in zip(columnas, METODOS.items()):
            resultado = buscador(pregunta_cmp, k=k)
            with col:
                st.subheader(nombre)
                st.dataframe(resultado[["id", columna_score(resultado)]], hide_index=True)
                for fila in resultado.head(3).itertuples():
                    with st.expander(fila.id):
                        st.write(fila.texto)

# --- Tab 3: evaluación con el golden dataset -----------------------
with tab_eval:
    st.markdown(
        "**Hit@k**: % de preguntas con algún fragmento correcto en el top-k. "
        "**MRR**: promedio de 1/posición del primer correcto (1 → 1.0, 2 → 0.5, no aparece → 0)."
    )
    st.dataframe(pd.DataFrame(GOLDEN), hide_index=True)

    if st.button("Correr evaluación", type="primary"):
        K = 5
        filas = []
        barra = st.progress(0.0)
        total = len(METODOS) * len(GOLDEN)
        for nombre, buscador in METODOS.items():
            for n, caso in enumerate(GOLDEN, start=1):
                ids = buscador(caso["pregunta"], k=K)["id"].tolist()
                filas.append({"metodo": nombre, "pregunta": f"P{n}",
                              "posicion": posicion_primer_relevante(ids, caso["relevantes"])})
                barra.progress(len(filas) / total)

        detalle = pd.DataFrame(filas)
        detalle["rr"] = detalle["posicion"].apply(lambda p: 0 if pd.isna(p) else 1 / p)

        resumen = detalle.groupby("metodo", sort=False).agg(
            **{
                "Hit@1": ("posicion", lambda p: (p <= 1).mean()),
                "Hit@3": ("posicion", lambda p: (p <= 3).mean()),
                "Hit@5": ("posicion", lambda p: (p <= 5).mean()),
                "MRR@5": ("rr", "mean"),
            }
        )

        st.subheader("Resumen")
        st.dataframe(resumen.style.format("{:.2f}"))
        st.bar_chart(resumen, stack=False)

        st.subheader("Posición del primer fragmento correcto (vacío = no está en el top-5)")
        posiciones = detalle.pivot(index="pregunta", columns="metodo", values="posicion")[list(METODOS)]
        st.dataframe(posiciones)
