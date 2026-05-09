"""
Carrega o modelo exclusivamente do MLflow Registry (DagsHub).

Não há fallback local para pickle.
"""

from __future__ import annotations

import concurrent.futures
import os
from functools import partial
from pathlib import Path

import mlflow
import mlflow.sklearn

DEFAULT_BINARY_THRESHOLD = 6.5


def mlflow_load_timeout_sec() -> float:
    return float(os.getenv("MLFLOW_MODEL_LOAD_TIMEOUT_SEC", "120"))


def _sklearn_load_from_registry(model_name: str, timeout_sec: float) -> tuple[object | None, str | None]:
    tracking_uri = (os.getenv("MLFLOW_TRACKING_URI") or "").strip()
    username = (os.getenv("DAGSHUB_USERNAME") or "").strip()
    token = (os.getenv("DAGSHUB_TOKEN") or "").strip()
    if not (tracking_uri and username and token):
        return None, (
            "Credenciais/URI ausentes. Defina MLFLOW_TRACKING_URI, "
            "DAGSHUB_USERNAME e DAGSHUB_TOKEN."
        )

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
                loaded = future.result(timeout=timeout_sec)
                return loaded, None
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
        return None, str(last_err)
    return None, "Modelo não encontrado no Registry."


def load_model_bundle(
    root: Path | None = None,
    *,
    model_path_override: str | None = None,
) -> tuple[dict | None, str]:
    """Retorna (bundle ou None, mensagem de origem / erro).

    Política estrita: somente MLflow Registry (@production).
    Sem fallback local para pickle.
    """
    _ = root or Path(__file__).resolve().parent
    _ = model_path_override
    timeout_sec = mlflow_load_timeout_sec()
    model_name = os.getenv("MLFLOW_MODEL_NAME", "wine-quality-binary")

    ml_pipeline, registry_err = _sklearn_load_from_registry(model_name, timeout_sec)
    if ml_pipeline is not None:
        return {
            "pipeline": ml_pipeline,
            "binary_threshold": DEFAULT_BINARY_THRESHOLD,
        }, "🌐 MLflow Registry (@production)"
    return None, (
        f"❌ Não foi possível carregar modelo do MLflow Registry (@production). "
        f"Detalhe: {registry_err or 'erro desconhecido'}. "
        f"Timeout configurado: {timeout_sec:g}s."
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
