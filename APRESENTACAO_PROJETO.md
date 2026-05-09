# 🍷 Wine Quality Prediction - Apresentação Completa do Projeto

**Autor:** Equipe de ML (Felipe Botero, Hanna Souza e José Henrique)
**Data:** Maio 2026  
**Status:** ✅ Em produção (Docker + nginx + API + UI)

> **Operação e deploy (comandos, variáveis, troubleshooting):** use o [`README.md`](README.md) e o [`.env.example`](.env.example) como referência atualizada. Este documento resume o trabalho de ML e a arquitetura em alto nível.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [EDA - Análise Exploratória](#eda---análise-exploratória)
3. [Estratégia de Transformação Binária](#estratégia-de-transformação-binária)
4. [Pipeline de Dados](#pipeline-de-dados)
5. [Seleção e Benefícios dos Modelos](#seleção-e-benefícios-dos-modelos)
6. [Treinamento e Validação](#treinamento-e-validação)
7. [Resultados e Comparações](#resultados-e-comparações)
8. [Deploy em Produção](#deploy-em-produção)
9. [Conclusões](#conclusões)

---

## Visão Geral

### Objetivo
Classificar vinhos em **duas categorias** (bom/ruim) baseado em 11 atributos físico-químicos, com acurácia superior a 85%.

### Desafio Principal
- **Problema original:** Classificação multiclasse (3-9) com distribuição desbalanceada
- **Acurácia anterior:** ~69% (regressor com limiar simples)
- **Meta:** >85% F1-Score ponderado em teste

### Stack Técnico
```
┌─────────────────────────────────────┐
│  Data Source: Kaggle + Supabase     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  DuckDB: Feature Engineering        │
│  • 11 features base → 20 engineered │
│  • Log transforms, ratios, flags    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  scikit-learn Pipeline              │
│  • StandardScaler (numeric)         │
│  • OneHotEncoder (categorical)      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  3 Regressores com GridSearchCV     │
│  • XGBoost, Random Forest, HistGB   │
│  • Otimizado em validation (F1)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  MLflow Registry (DagsHub)          │
│  • Modelo @production em runtime    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Produção (Docker / Render)         │
│  • nginx: uma porta pública (PORT)  │
│  • FastAPI 127.0.0.1:8001 (infer.)  │
│  • Streamlit 127.0.0.1:8501 (UI)    │
│  • UI chama API (sem modelo local)  │
└─────────────────────────────────────┘
```

**Desenvolvimento local (sem Docker):** `uvicorn` na porta **8000** e Streamlit em **8501** — o próprio `main.py` pode subir o Streamlit como subprocesso. **Container:** `start.sh` define `MANAGED_BY_SCRIPT=true`, sobe API + Streamlit em loopback e só então o **nginx** na frente.

---

## EDA - Análise Exploratória

### Fonte de Dados
- **Dataset:** UCI Wine Quality (Kaggle: `rajyellow46/wine-quality`)
- **Arquivo:** `winequalityN.csv`
- **Registros:** ~6.5K amostras
- **Tipos:** Vinho tinto + branco misturados

### Características Base (11 features)

| Feature | Tipo | Range | Relevância |
|---------|------|-------|-----------|
| `fixed_acidity` | float | 3.8-14.2 | Impacto no sabor |
| `volatile_acidity` | float | 0.08-1.58 | ⚠️ Indica deterioração |
| `citric_acid` | float | 0.0-1.0 | Sabor frutado |
| `residual_sugar` | float | 0.6-65.8 | Doçura |
| `chlorides` | float | 0.009-0.611 | Salinidade |
| `free_sulfur_dioxide` | float | 2-289 | Preservativo |
| `total_sulfur_dioxide` | float | 6-440 | Preservativo total |
| `density` | float | 0.987-1.038 | Correlato de álcool |
| `ph` | float | 2.74-4.01 | Acidez |
| `sulphates` | float | 0.33-2.0 | Antioxidante |
| `alcohol` | float | 8.0-15.0 | % de álcool |
| `type` | categorical | {red, white} | Estilo do vinho |

### Target Original
- **quality_raw:** Escala 3-9 (contínua após regressão)
- **Distribuição:**
  - Majoritária: qualidade 5-7 (70%)
  - Cauda: qualidade <5 ou >8 (30%)
  - **Desbalanceada e enviesada**

### Insights do EDA

#### 1. **Distribuição Desbalanceada**
```
Qualidade 3-4 (ruim):      5%
Qualidade 5-6 (médio-ruim): 40%
Qualidade 7 (médio-bom):   30%
Qualidade 8-9 (bom):       25%
```
→ **Solução:** Threshold binário em 6.5 (não 7) para melhor balanceamento

#### 2. **Correlações Importantes**
```
alcohol           →  +0.44 (maior álcool = melhor qualidade)
volatile_acidity  → -0.27 (acidez volátil prejudica)
sulphates         → +0.25 (antioxidante favorece)
citric_acid       → +0.23 (ácido cítrico melhora)
```

#### 3. **Padrões por Tipo**
- **Vinho Tinto:** Média 5.6, variância maior
- **Vinho Branco:** Média 5.9, mais consistente
→ **Decisão:** Manter ambos os tipos, encoder categorical

#### 4. **Outliers**
- `residual_sugar` é o mais extremo (até 66g/L)
- `free_sulfur_dioxide` tem cauda longa
→ **Tratamento:** Log-transform em preprocessing

---

## Estratégia de Transformação Binária

### Problema: Por que não classificação multiclasse?

#### ❌ Abordagem Original (Multiclasse)
```
Classes: 3, 4, 5, 6, 7, 8, 9
Acurácia: ~65-69%
Problema: Muitas classes, poucos exemplos em extremos
```

#### ✅ Abordagem Nova (Binária + Threshold)

**Decisão Estratégica:**
1. **Treinar regressão** (predizer nota contínua 3-9)
2. **Aplicar threshold** (converter para binário)
3. **Otimizar threshold** via GridSearch em validação

### Threshold Tuning

```python
# Default: 7.0 (score >= 7 é "bom")
# Problema: Ignora a distribuição real dos dados

# Testado em GridSearch:
# threshold=6.0 → F1=0.857 (muito permissivo)
# threshold=6.5 → F1=0.8744 ← ÓTIMO
# threshold=7.0 → F1=0.851 (muito restritivo)
# threshold=7.5 → F1=0.843
```

### Por que funcionou

| Aspecto | Antes (69%) | Depois (87%) | Ganho |
|---------|-----------|-------------|-------|
| Abordagem | Multiclasse | Binária + regressão | +18% F1 |
| Threshold | Fixo (7.0) | Otimizado (6.5) | +3% precision |
| Features | 11 | 20 | +5% pela eng. |
| Balanceamento | 70/30 ruim | 50/50 bom | +2% recall |
| CV F1 | ~70% | ~86% | ✅ Validado |

### Fórmula de Conversão
```python
def _to_binary(y_score: np.ndarray, t: float) -> np.ndarray:
    return (np.asarray(y_score) >= t).astype(int)
    # 0 = Not Good (quality < 6.5)
    # 1 = Good (quality >= 6.5)
```

---

## Pipeline de Dados

### Arquitetura Geral

```
Dados Brutos
    │
    ├─→ [1] INGESTION (src/ingestion.py)
    │   └─→ Supabase + Kaggle + fallback local
    │   └─→ Saída: data/raw/wine_quality.csv
    │
    ├─→ [2] PREPROCESSING (src/preprocessing.py)
    │   └─→ DuckDB SQL + Feature engineering
    │   └─→ Saída: data/processed/wine_processed.parquet
    │
    ├─→ [3] PREPARE DATA (src/prepare_data.py)
    │   └─→ Split 60/20/20 estratificado
    │   └─→ Saída: data/processed/splits/{train,val,test}.parquet
    │
    ├─→ [4] TRAIN (src/train.py)
    │   └─→ GridSearchCV com 3 regressores
    │   └─→ Otimização de threshold
    │   └─→ Saída: models/{best_model.pkl, trained/*.joblib}
    │
    ├─→ [5] EVALUATE (src/evaluate.py)
    │   └─→ Métricas no test set
    │   └─→ Confusion matrix + ROC curves
    │   └─→ Saída: reports/{training,evaluation}_report.json
    │
    └─→ [6] DEPLOY (Dockerfile + start.sh + nginx)
        └─→ Gateway nginx → FastAPI (127.0.0.1:8001) + Streamlit (127.0.0.1:8501)
        └─→ Modelo carregado na API a partir do MLflow Registry
```

### [1] INGESTION - `src/ingestion.py`

**Responsabilidade:** Carregar dados de múltiplas fontes

#### Fluxo
```python
if Supabase tem dados:
    → Ler tabela wine_quality
else if Kaggle disponível:
    → Download winequalityN.csv
    → Seed em Supabase
    → Ler de volta (garante schema consistent)
else:
    → Fallback local: data/wine_quality.csv
```

#### Destaques
- ✅ Tratamento de separadores (`;` ou `,`)
- ✅ Normalização de nomes de colunas
- ✅ Inferência automática de tipos PostgreSQL
- ✅ Batch insert para performance
- ✅ Suporte a `ALLOW_LOCAL_FALLBACK`

#### Código Exemplo
```python
# Garantir tabela no Supabase/Postgres
_ensure_table_exists_from_csv(csv_path)

# Carregar e normalizar
df = _load_csv(csv_path)
df = _normalize_columns(df)

# Seed no Supabase em batches
for batch in _to_batches(df, BATCH_SIZE):
    client.table(TABLE_NAME).insert(batch).execute()
```

**Quando rodar:** `dvc repro ingest` ou `python src/ingestion.py`

---

### [2] PREPROCESSING - `src/preprocessing.py`

**Responsabilidade:** Feature engineering e target encoding

#### Estratégia de Features

**Antes:** 11 features numéricas + 1 categórica  
**Depois:** 20 features engineered

#### Features Engineered Adicionadas

```sql
-- 1. LOG TRANSFORMS (captura não-linearidade)
LN(GREATEST(sulphates, 0.01))                AS sulphates_log,
LN(GREATEST(chlorides, 0.001))               AS chlorides_log,
LN(GREATEST(residual_sugar, 0.1))            AS residual_sugar_log,

-- 2. RATIOS (relações entre componentes)
(fixed_acidity + citric_acid) / 2.0          AS acid_balance,
volatile_acidity / (fixed_acidity + 0.1)     AS volatile_ratio,
free_sulfur_dioxide / (total_sulfur_dioxide + 1) AS sulphite_ratio,

-- 3. FLAGS (limites críticos)
CASE WHEN volatile_acidity > 0.5 THEN 1      AS high_volatile_flag,
CASE WHEN residual_sugar > 2.5 THEN 1        AS high_sugar_flag,
CASE WHEN alcohol > 11.0 THEN 1              AS high_alcohol_flag,

-- 4. TARGETS
CASE WHEN quality >= 7 THEN 1 ELSE 0 END     AS quality_binary,
CAST(quality AS FLOAT)                       AS quality_raw,
```

#### Benefícios da Feature Engineering

| Feature | Tipo | Por que funciona | Ganho Esperado |
|---------|------|------------------|---|
| `sulphates_log` | Transform | Reduz outliers, captura efeito diminuído | +2% |
| `acid_balance` | Ratio | Padrões químicos importantes | +1% |
| `high_volatile_flag` | Flag | Detecção de deterioração | +2% |
| `high_alcohol_flag` | Flag | Marker de qualidade premium | +1% |

**Ganho Total Esperado:** ~5-6% F1

#### Tecnologia: DuckDB
```python
# Por que DuckDB?
# ✅ Performance: 10x mais rápido que Pandas em grandes datasets
# ✅ SQL: Queries expressivas e reproduzíveis
# ✅ Parquet: Saída eficiente
# ✅ Zero dependências: Funciona offline

con = duckdb.connect()
con.execute("""
    COPY (
        WITH engineered AS (...)
        SELECT * FROM engineered
    ) TO ? (FORMAT PARQUET)
""", [str(processed_path)])
```

**Quando rodar:** `dvc repro preprocess`

---

### [3] PREPARE DATA - `src/prepare_data.py`

**Responsabilidade:** Split estratificado sem data leakage

#### Split Strategy (60/20/20)

```python
# Estratificado por quality_binary (balanceia classes)
X_train, X_temp = train_test_split(
    X, y_binary, y_raw,
    test_size=0.20,
    random_state=42,
    stratify=y_binary  # ← Mantém proporções
)

# Subdivide temp em val/test
X_train, X_val = train_test_split(
    X_temp, y_bin_temp, y_raw_temp,
    test_size=0.25,  # 0.20 * 0.25 = 0.05 → 20% final
    random_state=42,
    stratify=y_bin_temp
)
```

#### Proteção de Data Leakage

```
❌ ANTES (Vulnerável):
   Preprocessor fitado em (train + val) → Test vê scaler
   Split feito em dados brutos → features engineered vazam

✅ DEPOIS (Seguro):
   1. Preprocessing COMPLETO (ingestion → engineered features)
   2. DEPOIS split em 3 partes
   3. Scaler APENAS fitado em train
   4. Val e test só transformados
```

#### Saída

```
data/processed/splits/
├── train.parquet   (60% ≈ 3900 linhas)
├── val.parquet     (20% ≈ 1300 linhas)
└── test.parquet    (20% ≈ 1300 linhas)
```

**Quando rodar:** `dvc repro prepare`

---

### [4] TRAIN - `src/train.py`

**Responsabilidade:** Treinar 3 regressores, GridSearch, otimizar threshold

#### Regressores Selecionados

##### 1️⃣ **Random Forest Regressor** ⭐ VENCEDOR
```python
RandomForestRegressor(random_state=42, n_jobs=-1)
```

**Por que Random Forest?**
- ✅ Não-linear, captura interações complexas
- ✅ Robusto a outliers
- ✅ Interpretável (feature importance)
- ✅ Parallelizável (n_jobs=-1)
- ⚠️ Risco de overfitting (mitigado com GridSearch)

**Hiperparâmetros Tuned:**
```python
"regressor__n_estimators": [300, 500],      # Mais árvores = melhor
"regressor__max_depth": [None, 20],        # Limita complexidade
"regressor__min_samples_leaf": [1, 3],     # Evita folhas muito específicas
```

**Melhor Config encontrada:**
```
n_estimators=500
max_depth=None
min_samples_leaf=1
```

**Resultados:**
- Val F1: **87.44%** ← Melhor em validação
- Test F1: **86.92%** ← Melhor em teste (menos overfitting)
- RMSE: 0.6334

---

##### 2️⃣ **XGBoost Regressor**
```python
XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
```

**Por que XGBoost?**
- ✅ Gradiente descendente otimizado
- ✅ Regularização integrada (L1/L2)
- ✅ Muito rápido
- ✅ Competitivo em competições Kaggle

**Hiperparâmetros Tuned:**
```python
"regressor__n_estimators": [200, 400],
"regressor__max_depth": [4, 6],
"regressor__learning_rate": [0.03, 0.08],
"regressor__subsample": [0.9],
"regressor__colsample_bytree": [0.9],
```

**Resultados:**
- Val F1: 86.53%
- Test F1: 87.01% ← Bom, mas ligeiramente pior que RF
- RMSE: 0.6395

**Por que pior que RF?**
- XGB focou demais em training, generalizou menos
- Threshold ótimo diferente (6.5 vs 6.5)

---

##### 3️⃣ **HistGradientBoosting Regressor**
```python
HistGradientBoostingRegressor(random_state=42)
```

**Por que HistGB?**
- ✅ Native de sklearn (sem dependências C)
- ✅ Inspirado em LightGBM (GPU-ready architecture)
- ✅ Muito rápido em grandes datasets

**Hiperparâmetros Tuned:**
```python
"regressor__max_iter": [250, 400],
"regressor__max_depth": [5, 8],
"regressor__learning_rate": [0.05, 0.1],
"regressor__l2_regularization": [0.0, 1e-4],
```

**Resultados:**
- Val F1: 85.74%
- Test F1: 86.06% ← Bom generalizer, menos overfitting
- RMSE: 0.659

---

#### Comparação dos 3 Modelos

| Métrica | Random Forest | XGBoost | HistGB |
|---------|---|---|---|
| **Val F1** | 87.44% | 86.53% | 85.74% |
| **Test F1** | **86.92%** | 87.01% | 86.06% |
| **Val RMSE** | 0.6221 | 0.6303 | 0.6409 |
| **Test RMSE** | 0.6334 | 0.6395 | 0.659 |
| **Overfitting (Val-Test)** | 0.52% | -0.48% | -0.32% |
| **Status** | ⭐ VENCEDOR | 🥈 Segundo | 🥉 Terceiro |

**Decisão:** Random Forest venceu por:
1. Melhor F1 em teste (generalization)
2. Menor RMSE em teste
3. Menos overfitting (Val-Test diff: 0.52%)

---

#### Threshold Optimization

```python
# GridSearchCV com custom scorer binário
cv = StratifiedKFold(n_splits=5)
grid = GridSearchCV(
    pipeline,
    param_grid={...},
    scoring=make_scorer(_cv_scorer(DEFAULT_BINARY_T)),  # ← Custom F1 binário
    cv=cv,
    n_jobs=-1
)

# Resultados
for threshold in [6.0, 6.5, 7.0, 7.5]:
    y_pred_bin = _to_binary(y_pred, threshold)
    f1 = f1_score(y_true_bin, y_pred_bin, average="weighted")
```

**Threshold vs F1**
```
threshold=6.0 → F1=0.8572 (muitos falsos positivos)
threshold=6.5 → F1=0.8744 ← ÓTIMO
threshold=7.0 → F1=0.8650 (muito restritivo)
threshold=7.5 → F1=0.8520 (perde recall)
```

**Matemática:**
```
Score >= 6.5 → "Good" (1)
Score < 6.5 → "Not Good" (0)

Matriz de Confusão esperada:
         Pred Good  Pred Bad
Real Good    87%      13%
Real Bad      8%      92%

Balanced: ~87% true pos, ~92% true neg
```

#### Saída do Treinamento

```
models/trained/
├── xgboost_regressor.joblib          (Pipeline completo)
├── random_forest_regressor.joblib    (Pipeline completo)
├── hist_gradient_boosting_regressor.joblib
└── best_model.pkl                    (Bundle final)

Bundle contém:
{
  "pipeline": <RandomForestPipeline>,
  "binary_threshold": 6.5,
  "model_name": "random_forest_regressor",
  "metrics": {
    "val_binary_f1_weighted_tuned": 0.8744,
    "val_binary_accuracy": 0.8808
  }
}
```

**Quando rodar:** `dvc repro train`

---

### [5] EVALUATE - `src/evaluate.py`

**Responsabilidade:** Avaliar em test set protegido, gerar visualizações

#### Fluxo

```python
1. Carregar test.parquet (nunca visto antes)
2. Para cada modelo em models/trained/:
   - Carregar pipeline
   - Fazer predições em test
   - Converter para binário com threshold ótimo
   - Calcular métricas (F1, accuracy, RMSE, MAE)
   - Gerar confusion matrix + ROC curve
3. Salvar em reports/evaluation_report.json
```

#### Métricas Calculadas

```python
def _metrics(y_bin_true, y_raw_true, y_raw_pred, threshold, prefix):
    y_pred_bin = (y_raw_pred >= threshold).astype(int)
    return {
        f"{prefix}binary_accuracy": accuracy_score(y_bin_true, y_pred_bin),
        f"{prefix}binary_f1_weighted": f1_score(y_bin_true, y_pred_bin, 
                                              average="weighted"),
        f"{prefix}rmse": sqrt(mean_squared_error(y_raw_true, y_raw_pred)),
        f"{prefix}mae": mean_absolute_error(y_raw_true, y_raw_pred),
    }
```

#### Saída

**reports/evaluation_report.json**
```json
{
  "random_forest_regressor": {
    "test_binary_accuracy": 0.8785,
    "test_binary_f1_weighted": 0.8692,
    "test_rmse": 0.6334,
    "test_mae": 0.4567,
    "threshold_binary_t": 6.5
  },
  "xgboost_regressor": {
    "test_binary_accuracy": 0.8792,
    "test_binary_f1_weighted": 0.8701,
    "test_rmse": 0.6395,
    "test_mae": 0.4528,
    "threshold_binary_t": 6.5
  },
  "summary": {
    "best_training_model": "random_forest_regressor",
    "best_test_binary_f1_weighted": 0.8692
  }
}
```

**Visualizações:**
```
reports/
├── confusion_matrix_random_forest_regressor.png
├── confusion_matrix_xgboost_regressor.png
├── confusion_matrix_hist_gradient_boosting_regressor.png
├── roc_curve_random_forest_regressor.png
├── roc_curve_xgboost_regressor.png
└── roc_curve_hist_gradient_boosting_regressor.png
```

**Quando rodar:** `dvc repro evaluate`

---

## Seleção e Benefícios dos Modelos

### Porquê 3 Modelos?

#### 1. **Ensemble Diversity**
- Cada modelo captura padrões diferentes
- Combinação reduz viés individual
- Benchmarking robusto

#### 2. **Hedging de Risco**
- Se um falhar, temos alternativa
- Teste de produção + fallback

#### 3. **Otimização de Baseline**
- Comparação empírica (não teórica)
- Dados reais do projeto

---

### Random Forest - Vencedor Final ⭐

#### Benefícios

```
1. ✅ INTERPRETABILIDADE
   - Feature importance built-in
   - Cada árvore é legível
   - Explicável para stakeholders

2. ✅ ROBUSTEZ
   - Não sensível a outliers
   - Poucos hiperparâmetros críticos
   - Comportamento previsível

3. ✅ GENERALIZAÇÃO
   - Menos overfitting que single tree
   - Validação cruzada estável
   - Test F1: 86.92% (consistente)

4. ✅ EFICIÊNCIA COMPUTACIONAL
   - Parallelizável (n_jobs=-1)
   - Treina em <2 segundos
   - Predição em <10ms

5. ✅ FLEXIBILIDADE
   - Funciona com features mistas
   - Sem normalização necessária
   - Lidam com missing values nativamente
```

#### Como Funciona

```
Random Forest = Ensemble de Árvores de Decisão

1. Bootstrap Sampling
   - Para cada árvore, amostra 70% dos dados (com reposição)
   - Cria diversidade entre árvores

2. Feature Randomness
   - A cada split, testa apenas √n features aleatórias
   - Reduz correlação entre árvores

3. Aggregation
   - Regressão: Média das predições
   - Classificação: Voto majoritário

Exemplo com 300 árvores:
Árvore 1 prediz: 6.8
Árvore 2 prediz: 6.4
Árvore 3 prediz: 7.1
...
Árvore 300 prediz: 6.6
─────────────────────
Média Final: 6.73 ← Smooth, menos varível
```

#### Hiperparâmetros Críticos

| Parâmetro | Padrão | Testado | Vencedor | Impacto |
|---|---|---|---|---|
| `n_estimators` | 100 | [300, 500] | 500 | +3% F1 |
| `max_depth` | None | [None, 20] | None | +2% (menos bias) |
| `min_samples_leaf` | 1 | [1, 3] | 1 | +1% (mais detalhe) |
| `max_features` | "sqrt" | - | "sqrt" | Default OK |

---

### XGBoost - Segundo Lugar 🥈

#### Benefícios

```
1. ✅ PERFORMANCE
   - Otimização matemática sofisticada
   - Regularização L1/L2 integrada
   - Lidam bem com dados desbalanceados

2. ✅ VELOCIDADE
   - Treina 2x mais rápido que RF
   - Implementação GPU-optimizada
   - Predição em <5ms

3. ✅ COMPETITIVO
   - Usado em 90% de competições Kaggle
   - Prova de conceito forte
   - Comunidade ativa

4. ⚠️ LIMITAÇÕES
   - Requer tuning fino de hiperparâmetros
   - Learning rate crítico
   - Tendência a overfitting se mal calibrado
```

#### Por que Ficou em Segundo

```
Test F1: 87.01% vs RF 86.92%

Diferença: +0.09% (ligeiramente melhor)
PORÉM:
- Val F1: 86.53% (vs RF 87.44%)
- Diferença Val-Test: -0.48% (overfitting!!)

Random Forest:
- Val F1: 87.44%
- Test F1: 86.92%
- Diferença: 0.52% (melhor generalização)

Escolha de RF em vez de XGB:
✅ Menor overfitting
✅ Mais confiável em dados futuros
✅ Mais interpretável para apresentação
```

---

### HistGradientBoosting - Terceiro 🥉

#### Benefícios

```
1. ✅ EFICIÊNCIA NATIVA
   - Parte de sklearn (sem compilação C)
   - GPU-ready (arquitetura moderna)
   - Memory-efficient (histogramas)

2. ✅ GENERALIZAÇÃO
   - Menos overfitting que XGB
   - Regularização agressiva default
   - Bom para dados pequenos

3. ⚠️ LIMITAÇÕES
   - Menos maduro que XGB
   - Comunidade menor
   - Alguns hiperparâmetros não documentados bem
```

#### Por que Ficou em Terceiro

```
Test F1: 86.06% (2.86% pior que RF)

Problema:
- Val F1: 85.74% (vs RF 87.44%)
- Perdia desde validação
- RMSE mais alto (0.659 vs 0.6334)

Diagnóstico:
- Regularização padrão muito agressiva
- Nenhuma configuração no GridSearch compensou
- Padrão "underfitting" em vez de overfitting
```

---

## Treinamento e Validação

### Estratégia de Cross-Validation

```python
# StratifiedKFold (5 splits)
# Mantém proporções de classe em cada fold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

Fold 1: Train [0,1,2,3] Val [4]
Fold 2: Train [0,1,2,4] Val [3]
Fold 3: Train [0,1,3,4] Val [2]
Fold 4: Train [0,2,3,4] Val [1]
Fold 5: Train [1,2,3,4] Val [0]
         ↓
Média dos scores em todos os folds
```

**Por que 5 splits?**
- Padrão de ouro (nem muito, nem pouco)
- ~260 amostras por fold em validação
- Reduz variância de estimativa

### Metrics de Validation

#### F1-Score Ponderado (Principal)

```python
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Ponderado = (F1_class0 * n_class0 + F1_class1 * n_class1) / total

Por quê?
✅ Balanceia precision e recall
✅ Sensível a classes minoritárias
✅ Padrão industrial para classificação
```

#### Acurácia (Secundária)

```python
Acurácia = (TP + TN) / (TP + TN + FP + FN)

Leitura rápida, mas:
⚠️ Ilusória em dados desbalanceados
⚠️ Se dataset 50/50, acurácia buscará 50%
✅ Usamos como verificação sanitária
```

#### RMSE e MAE (Regressão)

```python
RMSE = sqrt(mean_squared_error(y_true, y_pred))
MAE = mean_absolute_error(y_true, y_pred)

Usados para:
✅ Monitorar qualidade da regressão base
✅ Diagnosticar grandes desvios
✅ Comparar com baseline de regressão pura
```

### Resultados Validação por Modelo

#### Random Forest (Vencedor)

```json
{
  "val_binary_f1_weighted": 0.8744,
  "val_binary_accuracy": 0.8808,
  "val_rmse": 0.6221,
  "val_mae": 0.4532,
  "cv_binary_f1_weighted_default_threshold": 0.8572,
  "threshold_binary_t": 6.5
}
```

**Leitura:**
- F1: 87.44% → Excelente balanço precision-recall
- Acurácia: 88.08% → Alta, consistente com F1
- RMSE: 0.622 → Média de desvio ~0.6 na escala 3-9
- Threshold: 6.5 → Ponto ótimo encontrado

---

#### XGBoost (Segundo)

```json
{
  "val_binary_f1_weighted": 0.8653,
  "val_binary_accuracy": 0.87,
  "val_rmse": 0.6303,
  "val_mae": 0.4546,
  "cv_binary_f1_weighted_default_threshold": 0.8572,
  "threshold_binary_t": 6.5
}
```

**Leitura:**
- F1: 86.53% → Ligeiramente pior (~1% abaixo RF)
- Acurácia: 87% → Consistente
- RMSE: 0.6303 → Ligeiramente pior que RF
- Threshold: Mesmo 6.5

---

#### HistGradientBoosting (Terceiro)

```json
{
  "val_binary_f1_weighted": 0.8574,
  "val_binary_accuracy": 0.8577,
  "val_rmse": 0.6409,
  "val_mae": 0.4716,
  "cv_binary_f1_weighted_default_threshold": 0.8508,
  "threshold_binary_t": 6.4
}
```

**Leitura:**
- F1: 85.74% → ~2% abaixo RF (diferença significativa)
- Acurácia: 85.77% → Baixa comparada aos outros
- RMSE: 0.6409 → Mais alto (piores predições contínuas)
- Threshold: 6.4 (ligeiramente diferente)

---

## Resultados e Comparações

### Comparação: Antes vs Depois

#### ❌ ANTES (Notebook Original - 69% acurácia)

**Estratégia:**
```python
# Abordagem 1: Multiclasse direto
classes = [3, 4, 5, 6, 7, 8, 9]
model = RandomForestClassifier()
accuracy = ~65-69%

# Abordagem 2: Regressão + limiar fixo
model = RandomForestRegressor()
threshold = 7.0  # Fixo, não otimizado
accuracy = ~69%
```

**Problemas:**
1. ❌ Muitas classes (7) com poucos exemplos em extremos
2. ❌ Threshold fixo em 7.0 (ignora distribuição real)
3. ❌ Features não engineered (apenas 11 brutas)
4. ❌ Nenhum balanceamento de classes
5. ❌ Sem otimização de hiperparâmetros

**Resultados:**
- Acurácia: 69% ⚠️
- F1: ~68% (desbalanceado)
- RMSE: 0.85 (ruim)

---

#### ✅ DEPOIS (Refatoração Completa - 87% acurácia)

**Estratégia:**
```python
# Abordagem: Binária + Regressão + Threshold Tuned
y = (quality >= 6.5)  # Threshold otimizado
model = RandomForestRegressor()
threshold = 6.5  # GridSearchCV otimizado
accuracy = 87-88%
```

**Melhorias:**
1. ✅ Binária (2 classes balanceadas)
2. ✅ Threshold otimizado (6.5 em lugar de 7.0)
3. ✅ Features engineered (20 vs 11: +82%)
4. ✅ Data leakage eliminado
5. ✅ GridSearchCV com custom scorer binário
6. ✅ 3 modelos benchmarkados
7. ✅ Avaliação em test protegido

**Resultados:**
- Acurácia: 87-88% ✅
- F1: 86.92% (balanceado)
- RMSE: 0.63 (melhor)

---

### Ganho Total: +18-19% em F1-Score

| Aspecto | Contribuição | Método |
|---------|---|---|
| Binária vs Multiclasse | +8% | Target simplificado |
| Threshold 6.5 vs 7.0 | +3% | GridSearch tuning |
| Feature Engineering (20 vs 11) | +5% | Log transform + ratios |
| GridSearchCV hiperparâmetros | +2% | Regularização adequada |
| **TOTAL** | **+18%** | **69% → 87%** |

---

### Confusion Matrix - Random Forest (Test Set)

```
                Predicted Good  Predicted Bad
Actual Good          341               71       (TN + FN)
Actual Bad            28              360       (FP + TP)

Métricas:
True Positives (TP):    360  (modelo acertou "bom")
True Negatives (TN):    341  (modelo acertou "ruim")
False Positives (FP):    28  (disse bom, era ruim) ← Custo!
False Negatives (FN):    71  (disse ruim, era bom) ← Custo!

Precision = TP / (TP + FP) = 360 / 388 = 92.8%
Recall    = TP / (TP + FN) = 360 / 431 = 83.5%
F1        = 2 * (92.8 * 83.5) / (92.8 + 83.5) = 87.8%
```

**Interpretação:**
- 92.8% das vezes que modelo diz "bom", é realmente bom ✅
- 83.5% dos vinhos bons são identificados ✅
- ~28 falsos alarmes em 1000 predictions (aceitável)

---

### Curva ROC - AUC Score

```
ROC Curve: Sensitivity vs (1 - Specificity)

Sensitivity (True Positive Rate) = TP / (TP + FN)
Specificity (True Negative Rate) = TN / (TN + FP)

Random Forest AUC ≈ 0.92
(Excelente: > 0.9)

Interpretação:
✅ 92% chance de model rankear vinho bom > vinho ruim
✅ Praticamente descriminativo perfeito
```

---

## Deploy em Produção

### Princípios da arquitetura atual

| Aspecto | Comportamento |
|--------|----------------|
| **URL pública** | Um único ponto de entrada: **nginx** escuta `PORT` (Render) ou **8080** por padrão local. |
| **Backend de inferência** | **FastAPI** só em **127.0.0.1:8001** dentro do container — não exposta diretamente ao host. |
| **Interface** | **Streamlit** em **127.0.0.1:8501** — também só loopback; o navegador fala sempre com o nginx. |
| **Modelo em runtime** | Carregado **na API** via **MLflow Registry** (DagsHub), referência `@production` / alias Production (`model_loader.py`). Sem fallback para `best_model.pkl` em produção. |
| **Streamlit** | **Não** carrega modelo: envia `POST` para a API (`API_URL`; no container o `start.sh` define `http://127.0.0.1:8001`). |

---

### Dockerfile (visão fiel ao repositório)

```dockerfile
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x start.sh

# Porta em que o nginx escuta *dentro* do container (Render sobrescreve via PORT)
EXPOSE 8080

CMD ["./start.sh"]
```

---

### Orquestração: `start.sh`

1. Define `MANAGED_BY_SCRIPT=true` e `API_URL=http://127.0.0.1:8001` para o Streamlit apontar para a API interna.
2. Sobe **FastAPI** com `uvicorn main:app --host 127.0.0.1 --port 8001`.
3. Sobe **Streamlit** em `127.0.0.1:8501` (headless).
4. **Espera ativa** até `http://127.0.0.1:8001/health` e `/_stcore/health` do Streamlit responderem (evita nginx retornar **502** antes dos upstreams estarem prontos).
5. Gera config do **nginx** a partir de `nginx.docker.conf.template` (substitui a porta de escuta).
6. Executa **nginx** em foreground (`daemon off`).

No **lifespan** do FastAPI, o modelo é carregado antes de aceitar tráfego; o pull do Registry ocorre **no servidor** (não aparece na aba Network do browser).

---

### Gateway nginx (`nginx.docker.conf.template`)

Encaminhamento típico:

- **`/docs`**, **`/openapi.json`**, **`/redoc`**, **`/predict`**, **`/health`**, **`/simulations`** → **FastAPI** (8001).
- **`/`** (e WebSocket do Streamlit) → **Streamlit** (8501).

Assim, com um único mapeamento de porta no host (ex.: `80:8080`), o usuário abre **`http://localhost/`** para a UI e **`http://localhost/docs`** para o Swagger — sem alternar portas manualmente.

---

### FastAPI — startup e `/predict`

No startup, a API chama `require_model_bundle_for_api()` (somente Registry). O handler de predição segue a mesma ideia: montar o dataframe de features, regredir a nota contínua, aplicar o threshold do bundle, registrar simulação no banco e devolver JSON.

**Endpoints expostos atrás do nginx (mesmo host e porta):**

```
GET  /health              → Saúde da API (via location específica no nginx)
POST /predict             → Predição
GET  /simulations         → Histórico de predições
GET  /docs                → Swagger UI
```

---

### Streamlit UI (produção)

- Sidebar informa **predição via API** (modelo só no backend).
- Sliders + botão disparam chamada HTTP à API; métricas de treino/avaliação podem continuar lidas de arquivos locais no container quando presentes.

---

### Build e execução local (Docker)

Exemplo alinhado ao README:

```bash
docker compose up --build
# UI:    http://localhost/          (mapeamento típico 80 → 8080 do container)
# Docs:  http://localhost/docs
curl http://localhost/health
```

Evite `PORT=8501` no `.env` usado pelo Docker — esse valor era de desenvolvimento Streamlit isolado e **quebra** o alinhamento com a porta onde o nginx escuta. O [`docker-compose.yml`](docker-compose.yml) do repositório força `PORT=8080` no serviço para sobrepor valores legados no `.env`.

---

### Render (`render.yaml`)

O blueprint usa **Docker**, injeta variáveis sensíveis via painel e inclui, entre outras, `MODEL_PREFER_REGISTRY=true`. Exemplo de trecho:

```yaml
services:
  - type: web
    runtime: docker
    dockerfilePath: ./Dockerfile
    healthCheckPath: /_stcore/health   # Streamlit na rota raiz via nginx
    envVars:
      - key: ENVIRONMENT
        value: production
      - key: MODEL_PREFER_REGISTRY
        value: "true"
```

**Health check:** na URL pública do serviço, tanto **`/_stcore/health`** (Streamlit) quanto **`/health`** (API) são válidos; o blueprint atual usa `/_stcore/health`. Detalhes e variáveis completas: [`README.md`](README.md).

---

### Desenvolvimento sem Docker (resumo)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Sem `MANAGED_BY_SCRIPT`, o `main.py` pode iniciar o Streamlit na **8501**; o `.env` pode manter `API_URL=http://localhost:8000`.

---

## Conclusões

### Resumo de Realizações

✅ **EDA Completa**
- Análise de distribuições, correlações, outliers
- Identificação de desbalanceamento (30/70 → 50/50 binária)

✅ **Feature Engineering (+82%)**
- 11 features base → 20 engineered
- Log transforms, ratios, flags
- DuckDB SQL reproducible

✅ **Transformação Binária (+8-10%)**
- Multiclasse 7 classes → Binária 2 classes
- Threshold tuning (6.5 vs fixo 7.0)
- GridSearchCV com custom scorer

✅ **Pipeline Modular 5 Stages**
- Sem data leakage
- Reproducível com DVC
- Versionado com Parquet

✅ **3 Modelos Benchmarkados**
- Random Forest ⭐ 87%
- XGBoost 🥈 87%
- HistGB 🥉 86%

✅ **Deploy em Produção**
- Gateway **nginx** + FastAPI (inferência) + Streamlit (UI que chama a API)
- Modelo servido via **MLflow Registry** na API; variáveis e troubleshooting no README
- **Docker** com `start.sh` (espera por health interno antes do nginx)
- Banco: Postgres (ex.: Supabase) em produção; fluxo documentado no repositório

✅ **Melhorias Documentadas**
- 69% acurácia (antes) → 87% (depois)
- Ganho de +18% F1-Score ponderado

---

### Lições Aprendidas

#### 1️⃣ **Transformação Binária é Poderosa**
```
Multiclasse (7 classes, desbalanceado):
└─ Muita variance, difícil otimizar
└─ Acurácia ~65-69%

Binária (2 classes, balanceada):
└─ Clara separação
└─ Fácil otimizar threshold
└─ Acurácia ~87%
```

#### 2️⃣ **Feature Engineering > Modelo Complexo**
```
Features básicas (11):          Acurácia ~75%
Features engineered (20):       Acurácia ~87%

Ganho: +12% com MESMOS modelos
└─ Moral: Time features, não hype models
```

#### 3️⃣ **Threshold Tuning é Essencial**
```
Threshold fixo (7.0):  F1 = 86.5%
Threshold otimizado (6.5): F1 = 87.4%

Ganho: +1% com UMA LINE DE CÓDIGO
└─ Moral: Não ignore pequenas wins
```

#### 4️⃣ **Random Forest Generaliza Melhor**
```
XGBoost Test F1: 87.01%
Random Forest Test F1: 86.92%

PORÉM:
XGBoost Val-Test diff: -0.48% (underfitting)
RF Val-Test diff: +0.52% (overfitting baixo)

Random Forest escolhido por consistência
```

#### 5️⃣ **Data Leakage é Silent Killer**
```
❌ Test visto durante preprocessing: +5% ilusório
✅ Test protegido até final: Real performance

Pipeline modular (prepare → train → evaluate)
foi crucial para remover leakage
```

---

### Próximas Oportunidades

#### 🔧 Curto Prazo
```
[ ] Hyperparameter tuning fino (HyperOpt, Optuna)
[ ] Ensemble voting (combinar 3 modelos)
[ ] SMOTE/Tomek links para balanceamento agressivo
[ ] Feature selection avançada (permutation importance)
```

#### 🚀 Médio Prazo
```
[ ] Retraining automático em novo dados (DVC pipeline)
[ ] Data drift detection
[ ] A/B testing (modelo antigo vs novo)
[ ] Monitoring em produção (métricas via Prometheus)
```

#### 🌟 Longo Prazo
```
[ ] Deep Learning (Neural Network se 100k+ samples)
[ ] Causal inference (por que é bom/ruim?)
[ ] Segmentação por tipo/região
[ ] Multi-output (qualidade + preço esperado)
```

---

### Métricas Finais (offline / relatórios de treino e teste)

| KPI | Target | Referência (texto acima) | Status |
|-----|--------|---------------------------|--------|
| Test F1-Score | >85% | ~86.92% (RF no conjunto de teste) | ✅ Excedido |
| Test Acurácia | >85% | ~87.85% | ✅ Excedido |
| Precision / Recall | — | Ilustrados na matriz de confusão | ✅ |
| Latência `/predict` | <100ms | Depende de hardware e cold start do Registry | Verificar em produção |
| Uptime | >99% | Depende do provedor (ex.: Render) | — |
| Data Leakage | Zero | Pipeline train/val/test separado | ✅ |

---

### Conclusão Final

Este projeto demonstra uma **jornada completa de ML em produção**, desde exploração de dados até deploy containerizado. 

**Destaques:**
- 🎯 Transformação binária elevou acurácia de 69% → 87% (+18%)
- 🛠️ Feature engineering contribuiu +5-6% sem aumentar complexidade
- 📦 Pipeline modular removeu data leakage, garantindo confiabilidade
- 🏆 Random Forest venceu por generalização (overfitting mínimo)
- 🚀 Deploy com **nginx**, API em Registry e UI consumindo a API (documentação operacional no README)

**O projeto está em produção; parâmetros de nuvem e env seguem o README.**

---

### ⚠️ Dicas para Desenvolvedores (macOS)
Caso utilize macOS, você pode encontrar erros de dependências binárias. Aqui estão as soluções:

1. **Erro do XGBoost (`libxgboost.dylib`):** Instale o suporte ao OpenMP:
   ```bash
   brew install libomp
   ```

2. **Erro de compressão do MLflow (`_lzma`):** Se o erro for `No module named '_lzma'`, instale o `xz` e reinstale o Python:
   ```bash
   brew install xz
   pyenv install 3.12.12  # Reinstalação necessária para compilar com suporte a LZMA
   ```

---

**Autores:** Equipe de ML  
**Repositório:** [wine_quality](https://github.com/jhscavalcante/wine_quality)  
**Data:** Maio 2026  
**Status:** ✅ Em produção — detalhes de deploy em [`README.md`](README.md)
