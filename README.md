# Asistente MercaAndes (Streamlit Community Cloud)

## Probar en local (opcional)
    pip install -r requirements.txt
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # pon tus claves
    streamlit run app.py

## Publicar en Streamlit Community Cloud
1. Sube estos archivos a un repo de GitHub (secrets.toml queda fuera por .gitignore).
2. Entra a https://share.streamlit.io con tu cuenta de GitHub → Create app → "Deploy a public app from GitHub".
3. Repository: tu repo · Branch: main · Main file path: app.py
4. Advanced settings → Python 3.11 → en Secrets pega:
       GEMINI_API_KEY = "..."
       QDRANT_URL = "https://xxxx.cloud.qdrant.io"
       QDRANT_API_KEY = "..."
5. Deploy. La primera vez tarda unos minutos (instala torch y descarga el re-ranker).
