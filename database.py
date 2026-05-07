"""
Conexão com banco de dados + health-check de startup.

Suporta PostgreSQL (produção) e SQLite (fallback local em Docker single-container).
"""

import os
import time
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./wine.db",
)

engine = SessionLocal = None  # type: ignore  # inicializados em connect()


def _is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite")

def _build_engine():
    global engine, SessionLocal
    if _is_sqlite_url(DATABASE_URL):
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def wait_for_db(retries: int = 15, delay: float = 2.0) -> None:
    """Bloqueia até o PostgreSQL aceitar conexões."""
    _build_engine()

    # SQLite local não precisa de retries/rede
    if _is_sqlite_url(DATABASE_URL):
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅  Banco SQLite disponível.")
        return

    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅  Banco de dados disponível.")
            return
        except Exception as exc:
            logger.warning(
                "⏳  Aguardando banco (%d/%d): %s", attempt, retries, exc
            )
            time.sleep(delay)
    raise RuntimeError("❌  Banco de dados não respondeu após %d tentativas." % retries)


class Base(DeclarativeBase):
    pass