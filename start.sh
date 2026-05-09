#!/bin/bash
set -e

echo "==> Iniciando Nginx (porta 8000)..."
nginx

echo "==> Iniciando Uvicorn (porta 8001)..."
exec uvicorn main:app --host 127.0.0.1 --port 8001
