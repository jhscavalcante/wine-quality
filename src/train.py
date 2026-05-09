"""
Módulo de treinamento dos modelos de predição de qualidade de vinho.

Estratégia: regressão + limiar de binarização ajustado.
  1. Treina três regressores (XGBoost, Random Forest, HistGradientBoosting)
     via GridSearchCV com validação cruzada estratificada.
  2. Para cada modelo, ajusta o melhor limiar de binarização no conjunto de
     validação (busca em grade de 6.0 a 7.1 em passos de 0.05).
  3. O modelo com o maior F1 ponderado tuned é salvo como `best_model.pkl`
     (bundle com pipeline + threshold + metadados).
  4. Todas as runs são rastreadas no MLflow (DagsHub ou local).
  5. O melhor modelo é automaticamente registrado no MLflow Model Registry
     com o alias `@production` para consumo da API e do script de avaliação.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]              # raiz do projeto
SPLITS_DIR = ROOT / "data" / "processed" / "splits"    # splits treino/val/teste
MODELS_DIR = ROOT / "models" / "trained"               # modelos individuais .joblib
APP_MODEL_PATH = ROOT / "models" / "best_model.pkl"    # bundle do melhor modelo
REPORT_PATH = ROOT / "reports" / "training_report.json" # métricas de validação

TARGET_BINARY = "quality_binary"  # label binário: 0=Not Good, 1=Good (>=7)
TARGET_RAW = "quality_raw"        # nota numérica original (3-9)
DEFAULT_BINARY_T = 6.5           # limiar padrão de binarização


def _setup_mlflow() -> None:
    """Configura o tracking do MLflow.

    Se as variáveis DAGSHUB_* estiverem definidas no .env, aponta para o
    servidor remoto do DagsHub. Caso contrário usa o diretório local `./mlruns`.
    """
    load_dotenv(ROOT / ".env")
    user = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "wine-quality-binary")

    if user and token and repo_name:
        # Tracking remoto no DagsHub
        dagshub_uri = f"https://dagshub.com/{user}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(dagshub_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    else:
        # Fallback: tracking local
        mlflow.set_tracking_uri("./mlruns")

    mlflow.set_experiment(experiment)


def _to_binary(y_score: np.ndarray, t: float) -> np.ndarray:
    """Converte scores contínuos em rótulos binários pelo limiar `t`."""
    return (np.asarray(y_score) >= t).astype(int)


def _cv_scorer(default_t: float):
    """Retorna uma função de scoring customizada para o GridSearchCV.

    Durante a validação cruzada, o target passado ao scorer é a nota numérica
    bruta (y_raw). A função binariza tanto o verdadeiro (limiar fixo 7) quanto
    o predito (limiar `default_t`) e retorna o F1 ponderado.
    """
    def _score(estimator, X, y_raw):
        y_pred_score = estimator.predict(X)
        y_true_bin = (np.asarray(y_raw) >= 7).astype(int)  # Good se nota >= 7
        y_pred_bin = _to_binary(y_pred_score, default_t)
        return float(f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0))

    return _score


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Constrói o pré-processador do pipeline.

    - Colunas numéricas → StandardScaler (normalização z-score).
    - Colunas categóricas (ex: 'type': red/white) → OneHotEncoder.
    - Colunas não identificadas são descartadas (`remainder='drop'`).
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )


def _regressors() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Retorna os regressores e seus grids de hiperparâmetros para busca.

    Cada entrada: nome → (instância do regressor, grid de hiperparâmetros).
    O prefixo 'regressor__' no grid refere-se ao passo 'regressor' do Pipeline.
    """
    return {
        "xgboost_regressor": (
            XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1),
            {
                "regressor__n_estimators": [200, 400],
                "regressor__max_depth": [4, 6],
                "regressor__learning_rate": [0.03, 0.08],
                "regressor__subsample": [0.9],
                "regressor__colsample_bytree": [0.9],
            },
        ),
        "random_forest_regressor": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "regressor__n_estimators": [300, 500],
                "regressor__max_depth": [None, 20],
                "regressor__min_samples_leaf": [1, 3],
            },
        ),
        "hist_gradient_boosting_regressor": (
            HistGradientBoostingRegressor(random_state=42),
            {
                "regressor__max_iter": [250, 400],
                "regressor__learning_rate": [0.03, 0.08],
                "regressor__max_depth": [None, 10],
                "regressor__min_samples_leaf": [20, 40],
            },
        ),
    }


def _tune_binary_threshold(y_binary_true: np.ndarray, y_pred_score: np.ndarray) -> tuple[float, float]:
    """Busca o limiar de binarização que maximiza o F1 ponderado no conjunto de validação.

    Testa limiares de 6.0 a 7.1 em passos de 0.05.
    Retorna (melhor_limiar, melhor_f1_ponderado).
    """
    best_t = DEFAULT_BINARY_T
    best_f1 = -1.0
    for t in np.round(np.arange(6.0, 7.1, 0.05), 2):
        y_pred_bin = _to_binary(y_pred_score, float(t))
        f1_bin = f1_score(y_binary_true, y_pred_bin, average="weighted", zero_division=0)
        if f1_bin > best_f1:
            best_f1 = float(f1_bin)
            best_t = float(t)
    return best_t, best_f1


def _metrics(y_bin_true: np.ndarray, y_raw_true: np.ndarray, y_raw_pred: np.ndarray, binary_t: float, prefix: str) -> dict[str, float]:
    y_pred_bin = _to_binary(y_raw_pred, binary_t)
    rmse = float(np.sqrt(np.mean((y_raw_true - y_raw_pred) ** 2)))
    mae = float(np.mean(np.abs(y_raw_true - y_raw_pred)))
    return {
        f"{prefix}binary_accuracy": round(float(accuracy_score(y_bin_true, y_pred_bin)), 4),
        f"{prefix}binary_f1_weighted": round(float(f1_score(y_bin_true, y_pred_bin, average="weighted", zero_division=0)), 4),
        f"{prefix}rmse": round(rmse, 4),
        f"{prefix}mae": round(mae, 4),
    }


def train() -> None:
    """Treina, ajusta e registra todos os modelos regressores.

    Fluxo principal:
    1. Configura MLflow e carrega splits de treino e validação.
    2. Cria bins quantílicos para permitir CV estratificada sobre regressão.
    3. Para cada regressor: busca por grid (GridSearchCV), ajusta limiar e
       registra métricas + modelo no MLflow.
    4. Salva todos os modelos individualmente em `models/trained/`.
    5. Persiste o melhor bundle global em `models/best_model.pkl`.
    6. Salva relatório de validação em `reports/training_report.json`.
    """
    _setup_mlflow()

    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val.parquet")

    X_train = train_df.drop(columns=[TARGET_BINARY, TARGET_RAW])
    y_train_raw = train_df[TARGET_RAW].values.astype(float)
    y_train_bin = train_df[TARGET_BINARY].values.astype(int)

    X_val = val_df.drop(columns=[TARGET_BINARY, TARGET_RAW])
    y_val_raw = val_df[TARGET_RAW].values.astype(float)
    y_val_bin = val_df[TARGET_BINARY].values.astype(int)

    # Divide as notas em 5 quantis para usar como estratos na CV
    # (StratifiedKFold exige classes discretas; usamos quantis da nota bruta)
    quantile_bins = pd.qcut(y_train_raw, q=5, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    report: dict[str, dict[str, float]] = {}  # acumula métricas por modelo
    best_global_f1 = -1.0                      # rastreia o melhor F1 global
    best_bundle: dict[str, Any] | None = None  # bundle do melhor modelo
    best_run_id: str | None = None             # run_id da melhor run no MLflow

    scorer = _cv_scorer(DEFAULT_BINARY_T)  # scorer customizado para o GridSearch

    for model_name, (regressor, param_grid) in _regressors().items():
        print(f"\n[train] Model: {model_name}")

        # Pipeline: pré-processamento → regressor
        pipe = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(X_train)),
                ("regressor", regressor),
            ]
        )

        # Busca pelo melhor conjunto de hiperparâmetros via CV estratificada
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv.split(X_train, quantile_bins),
            n_jobs=-1,   # usa todos os CPUs disponíveis
            verbose=0,
            refit=True,  # retreina o melhor estimador no dataset completo
        )

        gs.fit(X_train, y_train_raw)  # treina em y_raw (regressão)
        best_pipe: Pipeline = gs.best_estimator_

        y_val_pred_score = best_pipe.predict(X_val)  # scores no conjunto de validação
        # Ajusta o limiar de binarização maximizando F1 na validação
        binary_t, tuned_bin_f1 = _tune_binary_threshold(y_val_bin, y_val_pred_score)

        val_metrics = _metrics(y_val_bin, y_val_raw, y_val_pred_score, binary_t, prefix="val_")
        val_metrics["val_binary_f1_weighted_tuned"] = round(tuned_bin_f1, 4)              # F1 com threshold ajustado
        val_metrics["cv_binary_f1_weighted_default_threshold"] = round(float(gs.best_score_), 4)  # F1 médio da CV
        val_metrics["threshold_binary_t"] = round(binary_t, 2)                           # limiar escolhido

        report[model_name] = val_metrics

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_file = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(best_pipe, model_file)

        # Bundle: agrupa tudo que a API precisa para fazer predições
        bundle = {
            "type": "binary_regression_threshold_bundle",
            "version": 1,
            "pipeline": best_pipe,        # pipeline treinado (scaler + modelo)
            "binary_threshold": binary_t, # limiar otimizado
            "class_labels": {0: "Not Good", 1: "Good"},
            "target_raw": TARGET_RAW,
            "target_binary": TARGET_BINARY,
            "features": X_train.columns.tolist(),  # ordem das features esperada
            "model_name": model_name,
        }

        # Registra a run no MLflow com parâmetros, métricas e artefato do modelo
        with mlflow.start_run(run_name=f"binary_regression_threshold_{model_name}") as run:
            mlflow.log_param("model", model_name)
            mlflow.log_param("strategy", "binary_regression_plus_threshold")
            mlflow.log_param("threshold_binary_t", binary_t)
            for k, v in gs.best_params_.items():
                mlflow.log_param(k, str(v))  # registra cada hiperparâmetro escolhido
            mlflow.log_metrics(val_metrics)
            # Registra o modelo no MLflow Model Registry do DagsHub
            mlflow.sklearn.log_model(
                best_pipe,
                artifact_path="model",
                registered_model_name="wine-quality-binary",
            )
            current_run_id = run.info.run_id

        print(f"[train] val_binary_f1_weighted={val_metrics['val_binary_f1_weighted']} tuned={val_metrics['val_binary_f1_weighted_tuned']} t={binary_t:.2f}")

        if val_metrics["val_binary_f1_weighted_tuned"] > best_global_f1:
            best_global_f1 = val_metrics["val_binary_f1_weighted_tuned"]
            best_bundle = bundle
            best_run_id = current_run_id

    if best_bundle is None:
        raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

    APP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_bundle, APP_MODEL_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[train] ✅ Melhor bundle salvo em: {APP_MODEL_PATH}")
    print(f"[train] ✅ Relatório salvo em: {REPORT_PATH}")

    # Atribui o alias @production à versão do melhor modelo no MLflow Registry
    user = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    if best_run_id and user and token:
        try:
            from mlflow import MlflowClient
            client = MlflowClient()
            # Busca a versão do modelo registrada nesta run
            versions = client.search_model_versions(f"run_id='{best_run_id}'")
            if versions:
                best_version = versions[0].version
                client.set_registered_model_alias(
                    "wine-quality-binary", "production", best_version
                )
                print(f"[train] ✅ Alias @production → versão {best_version}")
            else:
                print("[train] ⚠️ Nenhuma versão encontrada para atribuir alias.")
        except Exception as exc:
            print(f"[train] ⚠️ Não foi possível atribuir alias @production: {exc}")


if __name__ == "__main__":
    train()
