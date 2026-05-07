"""
Tabela de log de simulações no PostgreSQL.
"""

from datetime import datetime

from sqlalchemy import Column, Integer, Float, DateTime, JSON
from sqlalchemy.dialects.postgresql import JSONB

try:
    from wine_project.database import Base
except Exception:
    from database import Base  # type: ignore


class SimulationLog(Base):
    __tablename__ = "simulation_logs"

    json_type = JSON().with_variant(JSONB, "postgresql")

    id                = Column(Integer, primary_key=True, index=True)
    created_at        = Column(DateTime, default=datetime.utcnow, nullable=False)
    predicted_quality = Column(Integer, nullable=False)
    predicted_score   = Column(Float, nullable=True)
    elapsed_ms        = Column(Float, nullable=False)
    input_data        = Column(json_type, nullable=False)   # features do vinho
    probabilities     = Column(json_type, nullable=False)   # distribuição de probabilidades
