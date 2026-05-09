"""
Registra cada predição na tabela `wine_predictions` do Supabase.

Schema alvo (binário):
- `predicted_quality`: 0=Not Good, 1=Good
- probabilidades: `prob_not_good`, `prob_good`

Observação:
- criação automática de tabela só funciona se houver URL Postgres com permissão DDL
    (SUPABASE_DATABASE_URL / SUPABASE_DB_URL / DATABASE_URL apontando para Supabase).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_client = None  # cliente Supabase inicializado preguiçosamente (lazy) na 1ª chamada
_last_error: str | None = None  # guarda o último erro para diagnóstico externo


def _get_client():
    global _client
    if _client is None:
        url = (os.getenv("SUPABASE_URL", "") or "").strip().strip('"').strip("'")
        key = (os.getenv("SUPABASE_KEY", "") or "").strip().strip('"').strip("'")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL e SUPABASE_KEY não configurados.")
        from supabase import create_client

        _client = create_client(url, key)
    return _client


SUPABASE_TABLE = os.getenv("SUPABASE_PREDICTIONS_TABLE", "wine_predictions")


def _clean_label(label: str) -> str:
    """Remove emojis de cor do label para armazenar apenas texto puro no banco."""
    return str(label).replace("🔴 ", "").replace("🟢 ", "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Conversão segura para float, retornando `default` em caso de falha."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_probabilities(probabilities: dict[str, float]) -> tuple[float, float]:
    """Mapeia probabilidades para o schema binário (not_good, good)."""
    clean = {_clean_label(k): _safe_float(v) for k, v in probabilities.items()}

    prob_not_good = (
        clean.get("Not Good")
        if "Not Good" in clean
        else clean.get("Ruim")
    )
    prob_good = (
        clean.get("Good")
        if "Good" in clean
        else clean.get("Bom")
    )

    if prob_not_good is None and prob_good is None:
        return 0.0, 0.0
    if prob_not_good is None:
        prob_not_good = 1.0 - _safe_float(prob_good)
    if prob_good is None:
        prob_good = 1.0 - _safe_float(prob_not_good)

    return _safe_float(prob_not_good), _safe_float(prob_good)


def _resolve_supabase_db_url() -> str:
    """Retorna URL Postgres para DDL da tabela de histórico (se disponível)."""
    candidates = [
        (os.getenv("SUPABASE_DATABASE_URL", "") or "").strip().strip('"').strip("'"),
        (os.getenv("SUPABASE_DB_URL", "") or "").strip().strip('"').strip("'"),
        (os.getenv("DATABASE_URL", "") or "").strip().strip('"').strip("'"),
    ]

    def _is_postgres_url(url: str) -> bool:
        return url.startswith("postgresql://") or url.startswith("postgres://")

    for url in candidates:
        if url and _is_postgres_url(url):
            return url
    return ""


def ensure_predictions_table() -> bool:
    """Tenta criar a tabela no Postgres do Supabase quando URL DDL está disponível."""
    db_url = _resolve_supabase_db_url()
    if not db_url or db_url.startswith("sqlite"):
        logger.info(
            "[supabase] Sem URL Postgres para DDL (SUPABASE_DATABASE_URL/SUPABASE_DB_URL)."
        )
        return False

    ddl = f'''
    CREATE TABLE IF NOT EXISTS public."{SUPABASE_TABLE}" (
        id                   bigserial PRIMARY KEY,
        created_at           timestamptz NOT NULL DEFAULT now(),
        fixed_acidity        numeric,
        volatile_acidity     numeric,
        citric_acid          numeric,
        residual_sugar       numeric,
        chlorides            numeric,
        free_sulfur_dioxide  numeric,
        total_sulfur_dioxide numeric,
        density              numeric,
        ph                   numeric,
        sulphates            numeric,
        alcohol              numeric,
        wine_type            text,
        predicted_quality    smallint NOT NULL,
        quality_label        text NOT NULL,
        prob_not_good        numeric NOT NULL DEFAULT 0,
        prob_good            numeric NOT NULL DEFAULT 0,
        elapsed_ms           numeric NOT NULL DEFAULT 0
    );
    '''

    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("[supabase] Tabela '%s' garantida.", SUPABASE_TABLE)
        return True
    except Exception as exc:
        logger.warning("[supabase] Falha ao criar tabela '%s': %s", SUPABASE_TABLE, exc)
        return False


def log_prediction(
    features: dict,
    predicted_quality: int,
    quality_label: str,
    probabilities: dict[str, float],
    elapsed_ms: float,
) -> bool:
    """
    Insere uma linha em `wine_predictions` no Supabase.
    Retorna True em sucesso, False em falha.
    """
    try:
        global _last_error
        prob_not_good, prob_good = _extract_probabilities(probabilities)

        row = {
            "fixed_acidity": _safe_float(features.get("fixed_acidity")),
            "volatile_acidity": _safe_float(features.get("volatile_acidity")),
            "citric_acid": _safe_float(features.get("citric_acid")),
            "residual_sugar": _safe_float(features.get("residual_sugar")),
            "chlorides": _safe_float(features.get("chlorides")),
            "free_sulfur_dioxide": _safe_float(features.get("free_sulfur_dioxide")),
            "total_sulfur_dioxide": _safe_float(features.get("total_sulfur_dioxide")),
            "density": _safe_float(features.get("density")),
            "ph": _safe_float(features.get("ph")),
            "sulphates": _safe_float(features.get("sulphates")),
            "alcohol": _safe_float(features.get("alcohol")),
            "wine_type": str(features.get("type", "")),
            "predicted_quality": int(predicted_quality),
            "quality_label": _clean_label(quality_label),
            "prob_not_good": prob_not_good,
            "prob_good": prob_good,
            "elapsed_ms": _safe_float(elapsed_ms),
        }
        _get_client().table(SUPABASE_TABLE).insert(row).execute()
        logger.info("[supabase] Predição registrada: quality=%d", predicted_quality)
        _last_error = None
        return True
    except Exception as exc:
        _last_error = str(exc)
        logger.warning("[supabase] Falha ao registrar predição: %s", exc)
        return False


def fetch_predictions(limit: int = 20) -> list[dict[str, Any]]:
    """Busca histórico direto do Supabase para uso no Streamlit."""
    try:
        resp = (
            _get_client()
            .table(SUPABASE_TABLE)
            .select("*")
            .order("created_at", desc=True)
            .limit(int(limit))
            .execute()
        )
        data = getattr(resp, "data", None) or []
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("[supabase] Falha ao buscar histórico: %s", exc)
        return []


def get_last_error() -> str:
    return _last_error or ""
