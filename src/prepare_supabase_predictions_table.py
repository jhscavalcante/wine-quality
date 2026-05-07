from __future__ import annotations

import sys
from datetime import datetime, UTC
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import supabase_logger

MARKER_PATH = ROOT / "reports" / "supabase_predictions_table_ready.txt"


def run() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)

    ok = supabase_logger.ensure_predictions_table()
    status = "ok" if ok else "failed"
    content = (
        f"status={status}\n"
        f"table={supabase_logger.SUPABASE_TABLE}\n"
        f"timestamp={datetime.now(UTC).isoformat()}\n"
        f"hint=configure SUPABASE_DB_URL/SUPABASE_DATABASE_URL with a reachable postgres URL\n"
    )
    MARKER_PATH.write_text(content, encoding="utf-8")

    if not ok:
        raise RuntimeError(
            "Não foi possível criar a tabela de predições no Supabase. "
            "Verifique SUPABASE_DB_URL/SUPABASE_DATABASE_URL e conectividade de rede."
        )

    print(f"[prepare_supabase_predictions_table] ✅ Tabela garantida: {supabase_logger.SUPABASE_TABLE}")
    print(f"[prepare_supabase_predictions_table] marker: {MARKER_PATH}")


if __name__ == "__main__":
    run()
