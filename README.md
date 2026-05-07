# Wine Quality Binary Classification

Projeto de ML com pipeline reproduzível para classificar vinhos em:
- `0 = Not Good` (`quality < 7`)
- `1 = Good` (`quality >= 7`)

A estratégia oficial é **regressão da nota (`quality_raw`) + threshold binário ajustado**.

## Stack
- Pipeline: DVC
- Tracking: MLflow (local ou DagsHub)
- ML: scikit-learn, XGBoost
- API: FastAPI
- UI: Streamlit

## Estrutura
- `src/ingestion.py`: carrega dados (Supabase com fallback local)
- `src/preprocessing.py`: cria `quality_raw` e `quality_binary`
- `src/prepare_data.py`: split 60/20/20 estratificado por `quality_binary`
- `src/train.py`: treino com GridSearch e ajuste de threshold binário
- `src/evaluate.py`: avaliação em teste + confusion matrix
- `main.py`: API `/predict`, `/health`, `/simulations`
- `streamlit_ui.py`: interface para predição e monitoramento

## Fonte de Dados (Source of Truth)

O projeto usa **Supabase como fonte de verdade operacional**.

Fluxo oficial:
1. Dataset Kaggle alvo: `rajyellow46/wine-quality` arquivo `winequalityN.csv`.
2. Executar `src/ingestion.py`, que:
   - tenta ler a tabela no Supabase (`SUPABASE_TABLE`);
   - se estiver vazia, baixa `winequalityN.csv` via Kaggle CLI, faz seed no Supabase e reconsulta;
   - salva snapshot local em `data/raw/wine_quality.csv`.
4. Executar `src/preprocessing.py`, que consome `data/raw/wine_quality.csv` com DuckDB e gera `data/processed/wine_processed.parquet`.

Resumo visual:
`Kaggle (winequalityN.csv) -> Supabase -> data/raw/wine_quality.csv -> DuckDB preprocessing -> data/processed/*`

Observações:
- Para download automático do Kaggle funcionar, configure `KAGGLE_USERNAME`/`KAGGLE_KEY` (ou `~/.kaggle/kaggle.json`) e tenha a CLI `kaggle` instalada.
- Se o Kaggle não estiver disponível, `ingestion.py` faz fallback para `data/wine_quality.csv`.

## Como executar

### 1) Ambiente
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Pipeline completo (recomendado)
```bash
dvc repro
```

### 3) Pipeline por etapas (manual)
```bash
python3 src/ingestion.py
python3 src/preprocessing.py
python3 src/prepare_data.py
python3 src/train.py
python3 src/evaluate.py
```

## Métricas oficiais
- Treino/validação: `reports/training_report.json`
  - Principal: `val_binary_f1_weighted_tuned`
- Teste: `reports/evaluation_report.json`
  - Principal: `test_binary_f1_weighted`

## API

### Health
```bash
GET /health
```

### Predição
```bash
POST /predict
```

Payload:
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "ph": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "type": "red"
}
```

Resposta (resumo):
- `quality`: 0 ou 1
- `quality_label`: `Not Good` ou `Good`
- `predicted_score`: nota contínua prevista
- `probabilities`: distribuição binária aproximada

## Notebooks
- `notebooks/wine_quality.ipynb`: execução e documentação do fluxo oficial
- `notebooks/wine_quality_with_grid_search.ipynb`: visão de tuning alinhada com `src/train.py`

## Observações
- O erro `"['type'] not in index"` ocorre quando a inferência faz one-hot manual de `type`; no fluxo atual isso foi removido e o encoder fica dentro do pipeline.
- Se algum arquivo de saída do DVC estiver rastreado no Git, remova com `git rm --cached <arquivo>` e comite.
