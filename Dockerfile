FROM python:3.12-slim

WORKDIR /app

# System deps (curl for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files and install deps
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Garantir que o script de inicialização é executável
RUN chmod +x start.sh

# Streamlit escuta em $PORT no Render (via start.sh); localmente costuma ser 8000 (docker run -p 8000:8000 -e PORT=8000)
EXPOSE 8000

# Inicia Streamlit (foreground) + FastAPI (background)
CMD ["./start.sh"]
