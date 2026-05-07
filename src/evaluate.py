from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
MODELS_DIR = ROOT / "models" / "trained"
BUNDLE_PATH = ROOT / "models" / "best_model.pkl"
TRAINING_REPORT_PATH = ROOT / "reports" / "training_report.json"
REPORT_PATH = ROOT / "reports" / "evaluation_report.json"

TARGET_BINARY = "quality_binary"
TARGET_RAW = "quality_raw"
CLASS_NAMES = ["Not Good(0)", "Good(1)"]
DEFAULT_BINARY_T = 6.5


def _to_binary(y_score: np.ndarray, t: float) -> np.ndarray:
    return (np.asarray(y_score) >= t).astype(int)


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


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
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
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    out = ROOT / "reports" / f"confusion_matrix_{name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def evaluate() -> None:
    test_df = pd.read_parquet(SPLITS_DIR / "test.parquet")
    X_test = test_df.drop(columns=[TARGET_BINARY, TARGET_RAW])
    y_test_bin = test_df[TARGET_BINARY].values.astype(int)
    y_test_raw = test_df[TARGET_RAW].values.astype(float)

    model_files = sorted(MODELS_DIR.glob("*_regressor.joblib"))
    if not model_files:
        model_files = sorted(MODELS_DIR.glob("*.joblib"))

    evaluation: dict[str, dict[str, float]] = {}

    for path in model_files:
        model = joblib.load(path)

        bundle: dict[str, Any] | None = None
        if BUNDLE_PATH.exists():
            maybe_bundle = joblib.load(BUNDLE_PATH)
            if isinstance(maybe_bundle, dict) and maybe_bundle.get("model_name") == path.stem:
                bundle = maybe_bundle

        binary_t = float(bundle.get("binary_threshold", DEFAULT_BINARY_T)) if bundle else DEFAULT_BINARY_T

        y_pred_raw = model.predict(X_test)
        y_pred_bin = _to_binary(y_pred_raw, binary_t)

        metrics = _metrics(y_test_bin, y_test_raw, y_pred_raw, binary_t, prefix="test_")
        metrics["threshold_binary_t"] = round(binary_t, 2)

        evaluation[path.stem] = metrics
        _save_confusion_matrix(y_test_bin, y_pred_bin, path.stem)

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
            pass

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"[evaluate] ✅ Relatório salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate()
