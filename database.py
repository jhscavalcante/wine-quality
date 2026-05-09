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

# Lê DATABASE_URL do ambiente; padrão SQLite para desenvolvimento local
DATABASE_URL = (os.getenv("DATABASE_URL", "sqlite:///./wine.db") or "").strip().strip('"').strip("'")
SQLITE_FALLBACK_URL = "sqlite:///./wine.db"  # banco local de emergência
# Se True, substitui DB remoto por SQLite caso a conexão falhe
AUTO_SQLITE_FALLBACK = (os.getenv("AUTO_SQLITE_FALLBACK", "true") or "true").lower() == "true"
# Número de tentativas de conexão na inicialização (0 = usa SQLite imediatamente)
DB_STARTUP_RETRIES = int((os.getenv("DB_STARTUP_RETRIES", "0") or "0").strip())
DB_STARTUP_DELAY_SEC = float((os.getenv("DB_STARTUP_DELAY_SEC", "2") or "2").strip())  # segundos entre tentativas

engine = SessionLocal = None  # type: ignore  # inicializados em connect()


def _is_sqlite_url(url: str) -> bool:
    return url.startswith("sqlite")

def _build_engine():
    """Cria o engine SQLAlchemy e a SessionLocal com base na DATABASE_URL atual.

    SQLite requer `check_same_thread=False` para funcionar com FastAPI async.
    """
    global engine, SessionLocal
    if _is_sqlite_url(DATABASE_URL):
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},  # necessário para SQLite + threads
        )
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def wait_for_db(retries: int | None = None, delay: float | None = None) -> None:
    """Bloqueia até o PostgreSQL aceitar conexões."""
    global DATABASE_URL
    retries = DB_STARTUP_RETRIES if retries is None else int(retries)
    delay = DB_STARTUP_DELAY_SEC if delay is None else float(delay)
    _build_engine()

    # SQLite local não precisa de retries/rede
    if _is_sqlite_url(DATABASE_URL):
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅  Banco SQLite disponível.")
        return

    if retries <= 0:
        if AUTO_SQLITE_FALLBACK:
            logger.warning(
                "⚠️  DB_STARTUP_RETRIES=%d. Pulando espera por banco remoto e "
                "aplicando fallback imediato para SQLite local.",
                retries,
            )
            DATABASE_URL = SQLITE_FALLBACK_URL
            _build_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("✅  Fallback SQLite ativado em %s", SQLITE_FALLBACK_URL)
            return
        raise RuntimeError(
            "❌  DB_STARTUP_RETRIES<=0 e AUTO_SQLITE_FALLBACK=false. "
            "Sem banco disponível para iniciar."
        )

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

    if AUTO_SQLITE_FALLBACK:
        logger.warning(
            "⚠️  Banco remoto indisponível após %d tentativas. "
            "Aplicando fallback automático para SQLite local.",
            retries,
        )
        DATABASE_URL = SQLITE_FALLBACK_URL
        _build_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅  Fallback SQLite ativado em %s", SQLITE_FALLBACK_URL)
        return

    raise RuntimeError("❌  Banco de dados não respondeu após %d tentativas." % retries)


class Base(DeclarativeBase):
    pass