"""
Carrega o bundle do modelo (pickle local ou MLflow Registry) com timeout no MLflow.

MODEL_PREFER_REGISTRY=true tenta o Registry antes do pickle (alinhado a promoções
@production no Render). Padrão false: pickle local primeiro. Timeout: MLFLOW_MODEL_LOAD_TIMEOUT_SEC.
"""

from __future__ import annotations

import concurrent.futures
import os
from functools import partial
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

DEFAULT_BINARY_THRESHOLD = 6.5


def prefer_registry_first() -> bool:
    """Se True, tenta MLflow @production antes do pickle (alinhado a promoções no Registry)."""
    v = (os.getenv("MODEL_PREFER_REGISTRY", "") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def mlflow_load_timeout_sec() -> float:
    return float(os.getenv("MLFLOW_MODEL_LOAD_TIMEOUT_SEC", "120"))


def _candidate_local_paths(root: Path, model_path_override: str | None) -> list[Path]:
    base = model_path_override or os.getenv(
        "MODEL_PATH", str(root / "models" / "best_model.pkl")
    )
    paths = [
        Path(base),
        root / "models" / "best_model.pkl",
        root / "best_model.pkl",
    ]
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _normalize_bundle(loaded: object) -> dict:
    if isinstance(loaded, dict) and "pipeline" in loaded:
        out = dict(loaded)
        out.setdefault("binary_threshold", DEFAULT_BINARY_THRESHOLD)
        return out
    return {"pipeline": loaded, "binary_threshold": DEFAULT_BINARY_THRESHOLD}


def _load_local_bundle(paths: list[Path]) -> dict | None:
    for model_path in paths:
        if not model_path.exists():
            continue
        try:
            loaded = joblib.load(model_path)
            return _normalize_bundle(loaded)
        except Exception:
            continue
    return None


def _sklearn_load_from_registry(model_name: str, timeout_sec: float) -> object | None:
    tracking_uri = (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    username = (os.getenv("DAGSHUB_USERNAME") or "").strip()
    token = (os.getenv("DAGSHUB_TOKEN") or "").strip()
    if not (tracking_uri and username and token):
        return None

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)

    refs = [
        f"models:/{model_name}@production",
        f"models:/{model_name}/Production",
    ]

    last_err: Exception | None = None
    for ref in refs:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(partial(mlflow.sklearn.load_model, ref))
                return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            last_err = TimeoutError(
                f"MLflow excedeu {timeout_sec}s ao carregar {ref}"
            )
            continue
        except Exception as e:
            last_err = e
            continue

    if last_err:
        print(f"[model_loader] MLflow Registry indisponível: {last_err}")
    return None


def load_model_bundle(
    root: Path | None = None,
    *,
    model_path_override: str | None = None,
) -> tuple[dict | None, str]:
    """Retorna (bundle ou None, mensagem de origem / erro).

    Ordem controlada por MODEL_PREFER_REGISTRY:
    - false (padrão): arquivo local → MLflow → local de novo.
    - true: MLflow → arquivo local (fallback se Registry/credenciais falharem).
    """
    root = root or Path(__file__).resolve().parent
    paths = _candidate_local_paths(root, model_path_override)
    timeout_sec = mlflow_load_timeout_sec()
    model_name = os.getenv("MLFLOW_MODEL_NAME", "wine-quality-binary")

    if prefer_registry_first():
        ml_pipeline = _sklearn_load_from_registry(model_name, timeout_sec)
        if ml_pipeline is not None:
            return {
                "pipeline": ml_pipeline,
                "binary_threshold": DEFAULT_BINARY_THRESHOLD,
            }, "🌐 MLflow Registry (@production)"

        local = _load_local_bundle(paths)
        if local is not None:
            used = next(p for p in paths if p.exists())
            return local, f"📁 Arquivo local (fallback): {used.name}"

        searched = ", ".join(str(p) for p in paths)
        return None, (
            f"❌ MLflow Registry indisponível e sem modelo local ({searched}). "
            f"Timeout {timeout_sec:g}s por tentativa ou credenciais/URI."
        )

    local = _load_local_bundle(paths)
    if local is not None:
        used = next(p for p in paths if p.exists())
        return local, f"📁 Arquivo local: {used.name}"

    ml_pipeline = _sklearn_load_from_registry(model_name, timeout_sec)
    if ml_pipeline is not None:
        return {
            "pipeline": ml_pipeline,
            "binary_threshold": DEFAULT_BINARY_THRESHOLD,
        }, "🌐 MLflow Registry (@production)"

    local2 = _load_local_bundle(paths)
    if local2 is not None:
        used = next(p for p in paths if p.exists())
        return local2, f"📁 Arquivo local: {used.name}"

    searched = ", ".join(str(p) for p in paths)
    return None, (
        f"❌ Modelo não encontrado ({searched}) "
        f"nem no MLflow Registry (timeout {timeout_sec:g}s ou credenciais/URI)."
    )


def require_model_bundle_for_api(
    root: Path | None = None,
    *,
    model_path_override: str | None = None,
) -> dict:
    bundle, msg = load_model_bundle(root, model_path_override=model_path_override)
    if bundle is None:
        raise RuntimeError(msg)
    return bundle
