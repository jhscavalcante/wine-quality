#!/bin/bash
set -e

# Sinaliza para o main.py NÃO iniciar o Streamlit como subprocesso
# (neste script, o próprio start.sh gerencia os dois processos)
export MANAGED_BY_SCRIPT=true
export API_URL=http://127.0.0.1:8001

# Render (e alguns PaaS) definem PORT; localmente usamos 8000 se não existir.
STREAMLIT_PORT="${PORT:-8000}"

echo "==> Iniciando FastAPI em background (porta 8001)..."
uvicorn main:app --host 127.0.0.1 --port 8001 &

echo "==> Iniciando Streamlit em foreground (porta ${STREAMLIT_PORT})..."
exec streamlit run streamlit_ui.py \
    --server.port="${STREAMLIT_PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
