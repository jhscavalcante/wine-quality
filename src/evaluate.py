"""
Módulo de avaliação final dos modelos treinados.

Carrega cada modelo salvo em `models/trained/`, faz predições sobre o
conjunto de teste (`data/processed/splits/test.parquet`) e calcula métricas
de classificação binária (acurácia, F1) e de regressão (RMSE, MAE).

Para cada modelo também é gerada uma matriz de confusão em PNG.
Ao final, o script tenta baixar o modelo oficial com alias `@production` do
MLflow Registry (DagsHub) para atualizar o bundle local `best_model.pkl`.
Um relatório JSON consolidado é salvo em `reports/evaluation_report.json`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Diretório raiz do projeto (dois níveis acima de src/evaluate.py)
ROOT = Path(__file__).resolve().parents[1]
# Diretório com os splits de treino/val/teste em formato Parquet
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
# Diretório com os modelos individuais salvos em .joblib
MODELS_DIR = ROOT / "models" / "trained"
# Caminho do bundle do melhor modelo (pipeline + threshold + metadados)
BUNDLE_PATH = ROOT / "models" / "best_model.pkl"
# Relatório gerado pelo treinamento (métricas de validação por modelo)
TRAINING_REPORT_PATH = ROOT / "reports" / "training_report.json"
# Relatório de saída desta etapa (métricas de teste por modelo)
REPORT_PATH = ROOT / "reports" / "evaluation_report.json"

# Nome das colunas-alvo no dataset processado
TARGET_BINARY = "quality_binary"   # coluna binária: 0=Not Good, 1=Good
TARGET_RAW = "quality_raw"         # coluna com a nota original (3-9)
CLASS_NAMES = ["Not Good(0)", "Good(1)"]  # rótulos para a matriz de confusão
DEFAULT_BINARY_T = 6.5             # limiar padrão: score >= 6.5 → classe Good
REGISTRY_MODEL_NAME = "wine-quality-binary"  # nome do modelo no MLflow Registry


def _setup_mlflow() -> bool:
    """Configura o MLflow para acessar o Registry no DagsHub.

    Carrega credenciais do .env e aponta o tracking URI.
    Retorna True se as credenciais estiverem disponíveis.
    """
    load_dotenv(ROOT / ".env")
    user = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "")

    if user and token and repo_name:
        dagshub_uri = f"https://dagshub.com/{user}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(dagshub_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        return True
    return False


def _load_production_model():
    """Baixa o modelo com alias @production do MLflow Registry.

    Retorna o pipeline sklearn carregado ou None se não for possível.
    """
    model_uri = f"models:/{REGISTRY_MODEL_NAME}@production"
    print(f"[evaluate] Tentando baixar modelo: {model_uri}")
    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
        print("[evaluate] ✅ Modelo @production carregado do Registry!")
        return pipeline
    except Exception as exc:
        print(f"[evaluate] ⚠️ Não foi possível baixar do Registry: {exc}")
        return None


def _to_binary(y_score: np.ndarray, t: float) -> np.ndarray:
    """Converte scores contínuos em rótulos binários usando o limiar `t`.
    Valores >= t recebem classe 1 (Good); abaixo recebem classe 0 (Not Good).
    """
    return (np.asarray(y_score) >= t).astype(int)


def _metrics(y_bin_true: np.ndarray, y_raw_true: np.ndarray, y_raw_pred: np.ndarray, binary_t: float, prefix: str) -> dict[str, float]:
    """Calcula e retorna um dicionário de métricas de avaliação.

    Métricas calculadas:
    - binary_accuracy  : acurácia da classificação binária (Good/Not Good)
    - binary_f1_weighted: F1-score ponderado pela distribuição das classes
    - rmse             : erro quadrático médio sobre a nota numérica
    - mae              : erro absoluto médio sobre a nota numérica

    O `prefix` diferencia métricas de treino ("val_") e teste ("test_").
    """
    y_pred_bin = _to_binary(y_raw_pred, binary_t)
    rmse = float(np.sqrt(np.mean((y_raw_true - y_raw_pred) ** 2)))
    mae = float(np.mean(np.abs(y_raw_true - y_raw_pred)))
    return {
        f"{prefix}binary_accuracy": round(float(accuracy_score(y_bin_true, y_pred_bin)), 4),
        f"{prefix}binary_f1_weighted": round(float(f1_score(y_bin_true, y_pred_bin, average="weighted", zero_division=0)), 4),
        f"{prefix}rmse": round(rmse, 4),
        f"{prefix}mae": round(mae, 4),
    }


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """Gera e salva a matriz de confusão como imagem PNG em `reports/`.

    Exibe os valores absolutos de cada célula (VP, VN, FP, FN) sobre o heatmap.
    A figura é fechada imediatamente para liberar memória.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    # Heatmap com escala de azul; valores mais altos = azul mais escuro
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True",
        xlabel="Pred",
        title=f"Confusion Matrix - {name}",
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    # Anota cada célula com o valor numérico correspondente
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    out = ROOT / "reports" / f"confusion_matrix_{name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)  # libera a figura da memória


def evaluate() -> None:
    """Avalia todos os modelos treinados sobre o conjunto de teste e salva o relatório.

    Fluxo:
    1. Carrega o conjunto de teste (X_test, y_test_bin, y_test_raw).
    2. Para cada modelo .joblib encontrado em `models/trained/`:
       a. Carrega o modelo e verifica se ele é o "melhor" pelo bundle salvo
          (para usar o limiar de binarização ajustado individualmente).
       b. Faz predições e calcula métricas de teste.
       c. Salva a matriz de confusão correspondente.
    3. Avalia também o bundle do melhor modelo global (`best_model.pkl`).
    4. Cria uma seção "summary" identificando qual modelo foi o melhor no treino.
    5. Persiste o relatório em `reports/evaluation_report.json`.
    """
    # --- 1. Carregar dados de teste ---
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")
    X_test = test_df.drop(columns=[TARGET_BINARY, TARGET_RAW])  # features
    y_test_bin = test_df[TARGET_BINARY].values.astype(int)       # label binário real
    y_test_raw = test_df[TARGET_RAW].values.astype(float)        # nota numérica real

    # --- 2. Listar arquivos de modelos treinados ---
    # Prefere padrão `*_regressor.joblib`; usa qualquer .joblib como fallback
    model_files = sorted(MODELS_DIR.glob("*_regressor.joblib"))
    if not model_files:
        model_files = sorted(MODELS_DIR.glob("*.joblib"))

    evaluation: dict[str, dict[str, float]] = {}

    # --- 3. Avaliar cada modelo individualmente ---
    for path in model_files:
        model = joblib.load(path)

        # Tenta usar o limiar ajustado do bundle, se este modelo for o melhor global
        bundle: dict[str, Any] | None = None
        if BUNDLE_PATH.exists():
            maybe_bundle = joblib.load(BUNDLE_PATH)
            if isinstance(maybe_bundle, dict) and maybe_bundle.get("model_name") == path.stem:
                bundle = maybe_bundle

        # Usa o limiar do bundle (ajustado na validação) ou o padrão 6.5
        binary_t = float(bundle.get("binary_threshold", DEFAULT_BINARY_T)) if bundle else DEFAULT_BINARY_T

        y_pred_raw = model.predict(X_test)          # predições numéricas (regressão)
        y_pred_bin = _to_binary(y_pred_raw, binary_t)  # binariza com o limiar

        metrics = _metrics(y_test_bin, y_test_raw, y_pred_raw, binary_t, prefix="test_")
        metrics["threshold_binary_t"] = round(binary_t, 2)  # registra o limiar usado

        evaluation[path.stem] = metrics
        _save_confusion_matrix(y_test_bin, y_pred_bin, path.stem)

    # --- 4. Avaliar o bundle do melhor modelo global ---
    # O bundle contém o pipeline completo (pré-processamento + regressão) e o
    # limiar de binarização otimizado na validação.
    if BUNDLE_PATH.exists():
        bundle = joblib.load(BUNDLE_PATH)
        if isinstance(bundle, dict) and "pipeline" in bundle:
            model = bundle["pipeline"]
            binary_t = float(bundle.get("binary_threshold", DEFAULT_BINARY_T))
            y_pred_raw = model.predict(X_test)
            y_pred_bin = _to_binary(y_pred_raw, binary_t)
            metrics = _metrics(y_test_bin, y_test_raw, y_pred_raw, binary_t, prefix="test_")
            metrics["threshold_binary_t"] = round(binary_t, 2)
            best_name = bundle.get("model_name", "best_bundle")
            evaluation[best_name] = metrics
            _save_confusion_matrix(y_test_bin, y_pred_bin, "best_bundle")

    # --- 5. Criar seção "summary" identificando o melhor modelo do treino ---
    # Lê o relatório de treinamento e identifica qual modelo teve o maior F1
    # ponderado na validação (com limiar ajustado, quando disponível).
    if TRAINING_REPORT_PATH.exists():
        try:
            train_report = json.loads(TRAINING_REPORT_PATH.read_text(encoding="utf-8"))
            best_train_model = max(
                train_report,
                key=lambda m: train_report[m].get("val_binary_f1_weighted_tuned", train_report[m].get("val_binary_f1_weighted", 0.0)),
            )
            evaluation["summary"] = {
                "best_training_model": best_train_model,
                "best_test_binary_f1_weighted": evaluation.get(best_train_model, {}).get("test_binary_f1_weighted", None),
            }
        except Exception:
            pass  # se o relatório de treino estiver corrompido, ignora o summary

    # --- 6. Persistir relatório de avaliação ---
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"[evaluate] ✅ Relatório salvo em: {REPORT_PATH}")

    # --- 7. Baixar modelo @production do Registry e salvar localmente ---
    # Isso garante que o best_model.pkl reflita o modelo de produção oficial.
    has_mlflow = _setup_mlflow()
    if has_mlflow:
        production_pipeline = _load_production_model()
        if production_pipeline is not None:
            # Constrói o bundle no formato esperado pela API
            prod_bundle = {
                "type": "binary_regression_threshold_bundle",
                "version": 1,
                "pipeline": production_pipeline,
                "binary_threshold": DEFAULT_BINARY_T,
                "class_labels": {0: "Not Good", 1: "Good"},
                "target_raw": TARGET_RAW,
                "target_binary": TARGET_BINARY,
            }
            # Usa o threshold do bundle local se disponível
            if BUNDLE_PATH.exists():
                local_bundle = joblib.load(BUNDLE_PATH)
                if isinstance(local_bundle, dict) and "binary_threshold" in local_bundle:
                    prod_bundle["binary_threshold"] = local_bundle["binary_threshold"]
                    prod_bundle["model_name"] = local_bundle.get("model_name", "production")
                    prod_bundle["features"] = local_bundle.get("features", [])
            BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(prod_bundle, BUNDLE_PATH)
            print(f"[evaluate] ✅ Bundle @production salvo em: {BUNDLE_PATH}")


if __name__ == "__main__":
    evaluate()
