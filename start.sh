#!/bin/bash
set -e

# Sinaliza para o main.py NÃO iniciar o Streamlit como subprocesso
# (neste script, o próprio start.sh gerencia os dois processos)
export MANAGED_BY_SCRIPT=true
export API_URL=http://127.0.0.1:8001

echo "==> Iniciando FastAPI em background (porta 8001)..."
uvicorn main:app --host 127.0.0.1 --port 8001 &

echo "==> Iniciando Streamlit em foreground (porta 8000)..."
exec streamlit run streamlit_ui.py \
    --server.port=8000 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
