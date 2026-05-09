FROM python:3.12-slim

WORKDIR /app

# System deps (curl + nginx)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Configuração do Nginx
COPY nginx.conf /etc/nginx/sites-available/default

# Expor API (8000) e Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Inicia a aplicação via FastAPI (que por sua vez inicia o Streamlit)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
