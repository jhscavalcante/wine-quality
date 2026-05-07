# GridSearch - Guia de Hiperparâmetros

## Overview
Este documento descreve os hiperparâmetros otimizados via GridSearch para cada modelo.

---

## 1. Logistic Regression

### Parâmetros Testados:
```python
{
    "classifier__C": [0.001, 0.01, 0.1, 1, 10],
    "classifier__penalty": ["l2"],
    "classifier__solver": ["lbfgs", "liblinear"],
}
```

**Explicação:**
- **C**: Inverso da força de regularização (menor C = regularização mais forte)
  - 0.001-0.01: Forte regularização (previne overfitting)
  - 1-10: Regularização fraca (permite maior complexidade)
  
- **penalty**: Tipo de regularização (L2 Ridge Regression)

- **solver**: Algoritmo de otimização
  - `lbfgs`: Melhor para datasets pequenos
  - `liblinear`: Mais rápido para datasets grandes

---

## 2. Random Forest

### Parâmetros Testados:
```python
{
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
}
```

**Explicação:**
- **n_estimators**: Número de árvores
  - Mais árvores = melhor performance, mas mais lento
  
- **max_depth**: Profundidade máxima das árvores
  - Menor profundidade = menos overfitting
  - `None` = sem limite (crescimento completo)
  
- **min_samples_split**: Mínimo de amostras para dividir um nó
  - Maior valor = árvores mais rasas, menos overfitting

---

## 3. K-Nearest Neighbors (KNN)

### Parâmetros Testados:
```python
{
    "classifier__n_neighbors": [3, 5, 7, 9, 11],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan"],
}
```

**Explicação:**
- **n_neighbors (k)**: Número de vizinhos considerados
  - Menor k = modelo mais complexo, mais overfitting
  - Maior k = modelo mais suave, possível underfitting
  
- **weights**: Peso dos vizinhos
  - `uniform`: Todos os vizinhos têm peso igual
  - `distance`: Vizinhos mais próximos têm mais peso
  
- **metric**: Medida de distância
  - `euclidean`: Distância Euclidiana (padrão)
  - `manhattan`: Distância de Manhattan (L1)

---

## 4. Support Vector Machine (SVM)

### Parâmetros Testados:
```python
{
    "classifier__C": [0.1, 1, 10],
    "classifier__kernel": ["linear", "rbf"],
    "classifier__gamma": ["scale", "auto"],
}
```

**Explicação:**
- **C**: Parâmetro de regularização
  - Menor C = maior margem, mais regularização
  - Maior C = menor margem, menor regularização
  
- **kernel**: Tipo de kernel
  - `linear`: Fronteira de decisão linear
  - `rbf`: Gaussian RBF (não-linear, mais poderoso)
  
- **gamma**: Coeficiente do kernel RBF
  - `scale`: 1 / (n_features * X.var())
  - `auto`: 1 / n_features

---

## 5. XGBoost

### Parâmetros Testados:
```python
{
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.01, 0.1, 0.3],
}
```

**Explicação:**
- **n_estimators**: Número de árvores de decisão
  - Mais árvores = modelo mais forte, mas mais lento
  
- **max_depth**: Profundidade máxima das árvores
  - Menor profundidade = menos overfitting
  - Recomendado: 3-7 para XGBoost
  
- **learning_rate**: Taxa de aprendizado
  - 0.01: Muito lento, melhor generalização
  - 0.1: Padrão balanceado
  - 0.3: Mais rápido, risco de overfitting

---

## 6. Multi-Layer Perceptron (MLP)

### Parâmetros Testados:
```python
{
    "classifier__hidden_layer_sizes": [(100,), (200, 100), (300, 150, 50)],
    "classifier__alpha": [0.0001, 0.001, 0.01],
    "classifier__learning_rate": ["constant", "adaptive"],
}
```

**Explicação:**
- **hidden_layer_sizes**: Arquitetura das camadas ocultas
  - (100,): 1 camada com 100 neurônios
  - (200, 100): 2 camadas (200 → 100 neurônios)
  - (300, 150, 50): 3 camadas (redução gradual)
  
- **alpha**: Parâmetro de regularização L2
  - Menor valor = menos regularização
  - Maior valor = mais regularização
  
- **learning_rate**: Estratégia da taxa de aprendizado
  - `constant`: Taxa fixa
  - `adaptive`: Reduz taxa quando plateauiza

---

## Combinações Testadas

Total de combinações por modelo:
```
Logistic Regression: 5 × 1 × 2 = 10
Random Forest:       3 × 3 × 3 = 27
KNN:                 5 × 2 × 2 = 20
SVM:                 3 × 2 × 2 = 12
XGBoost:             3 × 3 × 3 = 27
MLP:                 3 × 3 × 2 = 18
────────────────────────────────
TOTAL:                           114 modelos testados × 5 CV folds = 570 treinamentos
```

---

## Métrica de Seleção

- **Scoring**: `f1_weighted`
- **Cross-Validation**: 5 folds estratificados
- **Jobs**: `-1` (todos os cores disponíveis)

---

## Resultados Esperados

O GridSearch irá:
1. ✅ Treinar 570 modelos (114 × 5 folds)
2. ✅ Selecionar o melhor baseado em F1-Score
3. ✅ Registrar os melhores parâmetros no MLflow
4. ✅ Fornecer comparação com SMOTE e Tomek Links

---

## Dicas para Ajuste Futuro

- **Se modelo está underfitting**: Aumentar complexidade (maior `max_depth`, menos regularização)
- **Se modelo está overfitting**: Diminuir complexidade (menor `max_depth`, mais regularização)
- **Para acelerar**: Reduzir número de valores em cada parâmetro
- **Para melhor performance**: Aumentar número de folds (ex: 10 em vez de 5)

---

**Última atualização:** 2026-05-06
