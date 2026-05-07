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

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
MODELS_DIR = ROOT / "models" / "trained"
APP_MODEL_PATH = ROOT / "models" / "best_model.pkl"
REPORT_PATH = ROOT / "reports" / "training_report.json"

TARGET_BINARY = "quality_binary"
TARGET_RAW = "quality_raw"
DEFAULT_BINARY_T = 6.5


def _setup_mlflow() -> None:
    load_dotenv(ROOT / ".env")
    user = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "wine-quality-binary")

    if user and token and repo_name:
        dagshub_uri = f"https://dagshub.com/{user}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(dagshub_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    else:
        mlflow.set_tracking_uri("./mlruns")

    mlflow.set_experiment(experiment)


def _to_binary(y_score: np.ndarray, t: float) -> np.ndarray:
    return (np.asarray(y_score) >= t).astype(int)


def _cv_scorer(default_t: float):
    def _score(estimator, X, y_raw):
        y_pred_score = estimator.predict(X)
        y_true_bin = (np.asarray(y_raw) >= 7).astype(int)
        y_pred_bin = _to_binary(y_pred_score, default_t)
        return float(f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0))

    return _score


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
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
    _setup_mlflow()

    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val.parquet")

    X_train = train_df.drop(columns=[TARGET_BINARY, TARGET_RAW])
    y_train_raw = train_df[TARGET_RAW].values.astype(float)
    y_train_bin = train_df[TARGET_BINARY].values.astype(int)

    X_val = val_df.drop(columns=[TARGET_BINARY, TARGET_RAW])
    y_val_raw = val_df[TARGET_RAW].values.astype(float)
    y_val_bin = val_df[TARGET_BINARY].values.astype(int)

    quantile_bins = pd.qcut(y_train_raw, q=5, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    report: dict[str, dict[str, float]] = {}
    best_global_f1 = -1.0
    best_bundle: dict[str, Any] | None = None

    scorer = _cv_scorer(DEFAULT_BINARY_T)

    for model_name, (regressor, param_grid) in _regressors().items():
        print(f"\n[train] Model: {model_name}")

        pipe = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(X_train)),
                ("regressor", regressor),
            ]
        )

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv.split(X_train, quantile_bins),
            n_jobs=-1,
            verbose=0,
            refit=True,
        )

        gs.fit(X_train, y_train_raw)
        best_pipe: Pipeline = gs.best_estimator_

        y_val_pred_score = best_pipe.predict(X_val)
        binary_t, tuned_bin_f1 = _tune_binary_threshold(y_val_bin, y_val_pred_score)

        val_metrics = _metrics(y_val_bin, y_val_raw, y_val_pred_score, binary_t, prefix="val_")
        val_metrics["val_binary_f1_weighted_tuned"] = round(tuned_bin_f1, 4)
        val_metrics["cv_binary_f1_weighted_default_threshold"] = round(float(gs.best_score_), 4)
        val_metrics["threshold_binary_t"] = round(binary_t, 2)

        report[model_name] = val_metrics

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_file = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(best_pipe, model_file)

        bundle = {
            "type": "binary_regression_threshold_bundle",
            "version": 1,
            "pipeline": best_pipe,
            "binary_threshold": binary_t,
            "class_labels": {0: "Not Good", 1: "Good"},
            "target_raw": TARGET_RAW,
            "target_binary": TARGET_BINARY,
            "features": X_train.columns.tolist(),
            "model_name": model_name,
        }

        with mlflow.start_run(run_name=f"binary_regression_threshold_{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("strategy", "binary_regression_plus_threshold")
            mlflow.log_param("threshold_binary_t", binary_t)
            for k, v in gs.best_params_.items():
                mlflow.log_param(k, str(v))
            mlflow.log_metrics(val_metrics)
            mlflow.sklearn.log_model(best_pipe, artifact_path="model")

        print(f"[train] val_binary_f1_weighted={val_metrics['val_binary_f1_weighted']} tuned={val_metrics['val_binary_f1_weighted_tuned']} t={binary_t:.2f}")

        if val_metrics["val_binary_f1_weighted_tuned"] > best_global_f1:
            best_global_f1 = val_metrics["val_binary_f1_weighted_tuned"]
            best_bundle = bundle

    if best_bundle is None:
        raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

    APP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_bundle, APP_MODEL_PATH)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[train] ✅ Melhor bundle salvo em: {APP_MODEL_PATH}")
    print(f"[train] ✅ Relatório salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    train()
