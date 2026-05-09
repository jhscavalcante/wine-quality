#!/bin/bash
set -euo pipefail

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
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-120}"

echo "==> Gateway nginx em 0.0.0.0:${PUBLIC_LISTEN} → Streamlit:${INTERNAL_STREAMLIT_PORT} API:8001"

echo "==> Iniciando FastAPI em background (127.0.0.1:8001)..."
uvicorn main:app --host 127.0.0.1 --port 8001 &
API_PID=$!

echo "==> Iniciando Streamlit em background (127.0.0.1:${INTERNAL_STREAMLIT_PORT})..."
streamlit run streamlit_ui.py \
    --server.port="${INTERNAL_STREAMLIT_PORT}" \
    --server.address=127.0.0.1 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    &
STREAMLIT_PID=$!

wait_http() {
  local name="$1"
  local url="$2"
  local timeout="$3"
  local elapsed=0

  echo "==> Aguardando ${name} em ${url} (timeout: ${timeout}s)..."
  until curl -fsS "$url" >/dev/null 2>&1; do
    if ! kill -0 "$API_PID" >/dev/null 2>&1; then
      echo "❌ FastAPI encerrou antes de ficar pronto."
      exit 1
    fi
    if ! kill -0 "$STREAMLIT_PID" >/dev/null 2>&1; then
      echo "❌ Streamlit encerrou antes de ficar pronto."
      exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge "$timeout" ]; then
      echo "❌ Timeout aguardando ${name} em ${url}."
      exit 1
    fi
  done
  echo "✅ ${name} pronto."
}

# Evita 502 no nginx se os upstreams ainda não estiverem aceitando conexões
wait_http "FastAPI" "http://127.0.0.1:8001/health" "$STARTUP_TIMEOUT_SEC"
wait_http "Streamlit" "http://127.0.0.1:${INTERNAL_STREAMLIT_PORT}/_stcore/health" "$STARTUP_TIMEOUT_SEC"

sed "s/__LISTEN_PORT__/${PUBLIC_LISTEN}/g" "${TEMPLATE}" > "${NGINX_CONF}"
nginx -t -c "${NGINX_CONF}"

echo "==> Iniciando nginx (foreground)..."
exec nginx -c "${NGINX_CONF}" -g "daemon off;"
