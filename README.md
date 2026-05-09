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
1. Dataset Kaggle alvo: [rajyellow46/wine-quality](https://www.kaggle.com/datasets/rajyellow46/wine-quality) arquivo `winequalityN.csv`.
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

> **Notas para usuários macOS:**
> - **XGBoost:** Caso encontre erro ao carregar `libxgboost.dylib`, instale o OpenMP: `brew install libomp`.
> - **MLflow/LZMA:** Se encontrar erro `No module named '_lzma'`, instale `brew install xz` e reinstale sua versão do Python (ex: `pyenv install 3.12.12`).

### 2) Executar o pipeline de ML (recomendado)
Roda todas as etapas automaticamente via DVC (ingestão → pré-processamento → split → treino → avaliação):
```bash
dvc repro
```
> O DVC só re-executa etapas cujas dependências mudaram. Na primeira vez, executa tudo.

### 3) Pipeline por etapas (alternativa manual ao passo 2)
Execute cada script individualmente na ordem abaixo:
```bash
python3 src/ingestion.py          # baixa dados do Supabase/Kaggle → data/raw/
python3 src/preprocessing.py      # gera features e targets → data/processed/
python3 src/prepare_data.py       # split 60/20/20 → data/processed/splits/
python3 src/train.py              # treina modelos, salva best_model.pkl e relatório
python3 src/evaluate.py           # avalia no conjunto de teste, gera relatório + matriz de confusão
```

### 4) Verificar os resultados do treinamento

#### 4.1) Via DagsHub / MLflow Remoto (Recomendado)
Se você configurou as variáveis do DagsHub no `.env`, seus experimentos foram registrados na nuvem.
- Acesse a URL do seu repositório no DagsHub e clique na aba **MLflow**.
- Lá você encontrará todos os parâmetros, métricas e o gráfico de comparação entre os modelos treinados.

#### 4.2) Via Arquivos Locais e MLflow Local
Após o treinamento, os seguintes artefatos são gerados na sua máquina:

| Arquivo | Conteúdo |
|---|---|
| `reports/training_report.json` | Métricas de validação por modelo (F1, RMSE, MAE, threshold) |
| `reports/evaluation_report.json` | Métricas de teste por modelo + seção `summary` com o melhor |
| `reports/confusion_matrix_*.png` | Matrizes de confusão de cada modelo e do melhor bundle |
| `models/best_model.pkl` | Bundle do melhor modelo (pipeline + threshold + metadados) |

Para ver as métricas diretamente no terminal:
```bash
cat reports/training_report.json
cat reports/evaluation_report.json
```

Para visualizar as runs no MLflow Local (apenas se **não** estiver usando DagsHub):
```bash
mlflow ui --host 127.0.0.1 --port 5000
# Acesse: http://127.0.0.1:5000
```
> **⚠️ Nota:** Se você usa DagsHub, os registros de métricas e parâmetros são enviados apenas para o servidor remoto. Por isso, a interface do MLflow rodando localmente não mostrará essas execuções. Use sempre o IP `127.0.0.1` para evitar erro 403 em versões recentes do MLflow.

### 5) Testar a predição
Primeiro, suba a API FastAPI. Ela serve predições na porta 8000 e também inicializa o Streamlit automaticamente na porta 8501:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> Em ambas as opções abaixo, as predições são registradas automaticamente no **banco local** e no **Supabase**.

#### 5.1) Via Interface Streamlit (recomendado)
Acesse a interface gráfica no navegador:
```
http://localhost:8501
```
- Selecione o tipo de vinho (Tinto ou Branco).
- Ajuste os parâmetros físico-químicos com os sliders.
- Clique em **Classificar** para ver o resultado.

#### 5.2) Via API (Swagger ou curl)
Acesse a documentação interativa do Swagger no navegador:
```
http://localhost:8000/docs
```

Ou faça a chamada via `curl` no terminal:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Ver histórico de predições registradas:
```bash
curl http://localhost:8000/simulations
```

### 6) Subir via Docker (opcional)

**Pré-requisitos:**
1. Certifique-se de que o Docker Desktop (ou daemon do Docker) esteja em execução.
2. Pare a execução local (Passo 5) pressionando `Ctrl + C` no terminal antes de rodar o Docker para evitar conflito nas portas 8000 e 8501.

Para rodar tudo em container (API + Streamlit juntos):
```bash
docker build -t wine-quality .
docker run -p 8000:8000 -p 8501:8501 --env-file .env wine-quality
```

### 7) Publicar no GitHub
Crie um novo repositório no GitHub com o nome `wine-quality` e execute os comandos abaixo no terminal da raiz do projeto:

```bash
git init
git add .
git commit -m "app análise de qualidade de vinhos"
git branch -M main
git remote add origin https://github.com/<SEU_USUARIO>/wine-quality.git
git push -u origin main
```

### 8) Configurar o Web Service no Render

Siga os passos abaixo para hospedar sua aplicação (API + UI) no Render usando Docker:

1. **Acesse o Render:** Vá para [render.com](https://render.com) e faça login com sua conta do GitHub.
2. **Crie um Novo Web Service:**
   - Clique no botão **New** e selecione **Web Service**.
3. **Conecte o Repositório:**
   - Procure pelo seu repositório `wine-quality` e clique em **Connect**.
4. **Configurações Básicas:**
   - **Name:** Escolha um nome para seu serviço (ex: `wine-quality-api`).
   - **Runtime:** O Render deve detectar automaticamente o **Docker**. Caso contrário, selecione-o.
   - **Plan:** Selecione o plano **Free**.
5. **Configurar Variáveis de Ambiente:**
   - Clique em **Advanced** e depois em **Add Environment Variable**.
   - Adicione as chaves e valores presentes no seu arquivo `.env`. Para este projeto, as essenciais são:
     - `DATABASE_URL`: URL de conexão direta do Supabase (Postgres).
     - `SUPABASE_URL`: URL do projeto Supabase.
     - `SUPABASE_KEY`: Chave API (service_role ou anon) do Supabase.
     - `SUPABASE_TABLE`: Nome da tabela de dados (ex: `wine_quality`).
     - `SUPABASE_PREDICTIONS_TABLE`: Tabela para logs de predição (ex: `wine_predictions`).
     - `DAGSHUB_USERNAME`: Seu usuário no DagsHub.
     - `DAGSHUB_REPO_NAME`: Nome do repositório no DagsHub (`wine-quality`).
     - `DAGSHUB_TOKEN`: Seu token de acesso do DagsHub.
     - `MLFLOW_TRACKING_URI`: URI do MLflow (geralmente a do DagsHub).
6. **Publicação:**
   - Clique em **Create Web Service** ou **Deploy Web Service**.
7. **Verificação:**
   - O Render iniciará o build da imagem Docker. Quando o status mudar para **Live**, sua aplicação estará pública.
   - O próprio Render gerencia as portas baseando-se no `EXPOSE` do seu Dockerfile (geralmente a API na 8000 e o Streamlit na 8501, mas o Render costuma mapear a porta principal para o tráfego HTTP).

> [!TIP]
> **Resiliência do Modelo:** A aplicação está configurada para buscar o melhor modelo automaticamente no **MLflow Model Registry** do DagsHub usando o alias `@production`. Isso garante que, mesmo que o arquivo local `.pkl` não seja enviado para o repositório, a API conseguirá baixar a versão oficial de produção em tempo de execução.

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
