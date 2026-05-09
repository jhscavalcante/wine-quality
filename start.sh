#!/bin/bash
set -e

# Sinaliza para o main.py NÃO iniciar o Streamlit como subprocesso
export MANAGED_BY_SCRIPT=true
export API_URL=http://127.0.0.1:8001

APP_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_ROOT"

# Porta onde nginx escuta na rede do container (Render injeta PORT; local padrão 8080)
PUBLIC_LISTEN="${PORT:-8080}"
INTERNAL_STREAMLIT_PORT=8501
TEMPLATE="${APP_ROOT}/nginx.docker.conf.template"
NGINX_CONF="/tmp/nginx-proxy.conf"

echo "==> Gateway nginx em 0.0.0.0:${PUBLIC_LISTEN} → Streamlit:${INTERNAL_STREAMLIT_PORT} API:8001"

echo "==> Iniciando FastAPI em background (127.0.0.1:8001)..."
uvicorn main:app --host 127.0.0.1 --port 8001 &

echo "==> Iniciando Streamlit em background (127.0.0.1:${INTERNAL_STREAMLIT_PORT})..."
streamlit run streamlit_ui.py \
    --server.port="${INTERNAL_STREAMLIT_PORT}" \
    --server.address=127.0.0.1 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    &

sleep 2

sed "s/__LISTEN_PORT__/${PUBLIC_LISTEN}/g" "${TEMPLATE}" > "${NGINX_CONF}"
nginx -t -c "${NGINX_CONF}"

echo "==> Iniciando nginx (foreground)..."
exec nginx -c "${NGINX_CONF}" -g "daemon off;"
