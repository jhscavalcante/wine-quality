FROM python:3.12-slim

WORKDIR /app

# System deps (curl + nginx)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Configuração do Nginx (proxy reverso na porta 8000)
COPY nginx.conf /etc/nginx/sites-available/default

# Garantir que o script de inicialização é executável
RUN chmod +x start.sh

# Porta pública (Nginx)
EXPOSE 8000

# Inicia Nginx + Uvicorn via script
CMD ["./start.sh"]
