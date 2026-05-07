#!/bin/bash
# Script para executar o notebook com GridSearch

echo "🚀 Iniciando notebook com GridSearch..."
echo ""

# Verificar se estamos no diretório correto
if [ ! -f "notebooks/wine_quality_with_grid_search.ipynb" ]; then
    echo "❌ Erro: Notebook não encontrado!"
    echo "   Execute este script no diretório raiz do projeto"
    exit 1
fi

# Ativar virtual environment (se existir)
if [ -d ".venv" ]; then
    echo "✅ Ativando virtual environment..."
    source .venv/bin/activate
fi

# Verificar se jupyter está instalado
if ! command -v jupyter &> /dev/null; then
    echo "⚠️  Jupyter não encontrado. Instalando..."
    pip install jupyter
fi

# Iniciar jupyter notebook
echo ""
echo "📓 Abrindo Jupyter Notebook..."
echo "   URL: http://localhost:8888"
echo ""
echo "💡 Dicas:"
echo "   1. Execute as células na ordem (1 a 48)"
echo "   2. Células 15-22 são o novo GridSearch"
echo "   3. Verifique os resultados em Célula 20 (comparação)"
echo "   4. Logs estão em MLflow: mlflow ui"
echo ""

jupyter notebook notebooks/wine_quality_with_grid_search.ipynb
