"""
Wine Quality Prediction API + Streamlit UI (single container)

FastAPI roda na porta 8000.
Streamlit é iniciado como subprocesso na porta 8501 no startup.
"""

import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

try:
    import wine_project.database as database
    from wine_project.models import Base, SimulationLog
    import wine_project.supabase_logger as supabase_logger
except Exception:
    import database  # type: ignore
    from models import Base, SimulationLog  # type: ignore
    import supabase_logger  # type: ignore


def _load_env_files() -> None:
    """Carrega .env de caminhos comuns (local/dev e container)."""
    candidates = [
        Path(__file__).resolve().parent / ".env",  # /app/.env (container) ou root local
        Path.cwd() / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


_load_env_files()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv(
    "MODEL_PATH", str(Path(__file__).resolve().parent / "models" / "best_model.pkl")
)
STREAMLIT_UI = Path(__file__).resolve().parent / "streamlit_ui.py"

# Classes: 0=Not Good (<7), 1=Good (>=7)
CLASS_LABELS = {0: "Not Good", 1: "Good"}
DEFAULT_BINARY_THRESHOLD = 6.5

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model():
    """Carrega o modelo via MLflow Registry ou fallback local (joblib).

    Preferência:
    1. MLflow Registry remoto: tenta alias @champion, depois stage Production.
    2. Arquivo local `best_model.pkl` (bundle com pipeline + threshold).
    3. Qualquer arquivo .pkl encontrado nos caminhos candidatos.

    Sempre retorna um dict com chaves 'pipeline' e 'binary_threshold'.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    username = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "wine-quality-binary")

    if tracking_uri and username and token:
        try:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_tracking_uri(tracking_uri)
            # Try alias @champion first (MLflow >=2.9), fallback to stage Production
            for ref in [
                f"models:/{model_name}@production",
                f"models:/{model_name}/Production",
            ]:
                try:
                    loaded = mlflow.sklearn.load_model(ref)
                    return {"pipeline": loaded, "binary_threshold": DEFAULT_BINARY_THRESHOLD}
                except Exception:
                    continue
        except Exception as exc:
            print(f"[main] MLflow falhou ({exc}), usando fallback local.")

    # Caminhos candidatos para o bundle local (container e desenvolvimento local)
    candidate_paths = [
        Path(MODEL_PATH),
        Path(__file__).resolve().parent / "models" / "best_model.pkl",
        Path(__file__).resolve().parent / "best_model.pkl",
    ]

    for model_path in candidate_paths:
        if model_path.exists():
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and "pipeline" in loaded:
                return loaded
            return {"pipeline": loaded, "binary_threshold": DEFAULT_BINARY_THRESHOLD}

    searched = ", ".join(str(p) for p in candidate_paths)
    raise RuntimeError(
        f"Modelo não encontrado ({searched}) nem no MLflow Registry."
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

_streamlit_proc: subprocess.Popen | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação FastAPI.

    Na inicialização (before yield):
      - Aguarda o banco de dados estar disponível e cria as tabelas.
      - Carrega o modelo ML no estado compartilhado `app.state.model`.
      - Inicializa o Streamlit como subprocesso na porta 8501.

    No encerramento (after yield):
      - Termina o processo do Streamlit graciosamente.
    """
    global _streamlit_proc

    # Banco
    database.wait_for_db()
    Base.metadata.create_all(bind=database.engine)

    # Modelo
    app.state.model = _load_model()

    # Streamlit como subprocesso
    if STREAMLIT_UI.exists():
        _streamlit_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(STREAMLIT_UI),
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--server.headless=true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[main] Streamlit iniciado (PID={_streamlit_proc.pid}) na porta 8501")
    else:
        print(
            f"[main] streamlit_ui.py não encontrado em {STREAMLIT_UI}, UI desabilitada."
        )

    yield  # a aplicação fica em execução aqui

    if _streamlit_proc is not None:
        _streamlit_proc.terminate()
        _streamlit_proc.wait()
        print("[main] Streamlit encerrado.")


app = FastAPI(
    title="Wine Quality API",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., examples=[7.4])
    volatile_acidity: float = Field(..., examples=[0.70])
    citric_acid: float = Field(..., examples=[0.00])
    residual_sugar: float = Field(..., examples=[1.9])
    chlorides: float = Field(..., examples=[0.076])
    free_sulfur_dioxide: float = Field(..., examples=[11.0])
    total_sulfur_dioxide: float = Field(..., examples=[34.0])
    density: float = Field(..., examples=[0.9978])
    ph: float = Field(..., examples=[3.51])
    sulphates: float = Field(..., examples=[0.56])
    alcohol: float = Field(..., examples=[9.4])
    type: str = Field(..., examples=["red"], description="Wine type: 'red' or 'white'")


class PredictionResponse(BaseModel):
    quality: int
    quality_label: str
    probabilities: dict[str, float]
    predicted_score: float
    elapsed_ms: float


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Feature engineering — espelha preprocessing.py (11 features + type categórica)
# ---------------------------------------------------------------------------

FEATURE_ORDER_RAW = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
    "type",
]


def _get_expected_features(model) -> list[str]:
    if isinstance(model, dict) and "features" in model:
        return list(model["features"])
    for obj in [
        model,
        getattr(model, "steps", [[None, None]])[0][1]
        if hasattr(model, "steps")
        else None,
    ]:
        if obj is not None and hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
    try:
        return list(model.named_steps["preprocessor"].transformers_[0][2])
    except Exception:
        return FEATURE_ORDER_RAW


def _build_dataframe(wine: WineFeatures, model=None) -> pd.DataFrame:
    data = wine.model_dump()
    df = pd.DataFrame([data])

    if model is not None:
        expected = _get_expected_features(model)
        return df[expected]
    return df[FEATURE_ORDER_RAW]


def _score_to_class(score: float, binary_threshold: float) -> int:
    return 1 if float(score) >= float(binary_threshold) else 0


def _score_to_probabilities(score: float) -> dict[str, float]:
    """Converte um score numérico em probabilidades aproximadas por classe.

    Usa distância dos centros das classes (5.6=Not Good, 7.2=Good) como logits
    negativos, depois normaliza via softmax. Não é uma probabilidade calibrada,
    mas fornece uma representação intuitiva da confiança da predição.
    """
    centers = np.array([5.6, 7.2], dtype=float)  # centros típicos de cada classe
    dist = np.abs(centers - float(score))         # distância do score a cada centro
    logits = -dist                                 # mais perto = logit maior
    exps = np.exp(logits - np.max(logits))        # softmax numéricamente estável
    probs = exps / np.sum(exps)
    return {
        CLASS_LABELS[0]: round(float(probs[0]), 4),
        CLASS_LABELS[1]: round(float(probs[1]), 4),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["infra"])
def health_check():
    """Liveness probe."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(wine: WineFeatures, db: Session = Depends(get_db)):
    """Prediz a qualidade do vinho (0=Not Good, 1=Good).

    Fluxo:
    1. Extrai o pipeline e o threshold do bundle carregado no lifespan.
    2. Monta o DataFrame com as features na ordem correta.
    3. Realiza a predição (score contínuo) e binariza pelo threshold.
    4. Persiste o log no banco local (SQLite/Postgres) e no Supabase.
    5. Retorna score, classe, probabilidades e tempo de resposta.
    """
    model_bundle = app.state.model
    model = model_bundle["pipeline"] if isinstance(model_bundle, dict) else model_bundle
    binary_threshold = float(model_bundle.get("binary_threshold", DEFAULT_BINARY_THRESHOLD)) if isinstance(model_bundle, dict) else DEFAULT_BINARY_THRESHOLD
    t0 = time.perf_counter()

    df = _build_dataframe(wine, model_bundle)

    try:
        predicted_score = float(model.predict(df)[0])
        quality = _score_to_class(predicted_score, binary_threshold)
        probabilities = _score_to_probabilities(predicted_score)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {exc}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    log = SimulationLog(
        input_data=wine.model_dump(),
        predicted_quality=quality,
        probabilities=probabilities,
        predicted_score=predicted_score,
        elapsed_ms=elapsed_ms,
    )
    db.add(log)
    db.commit()

    # Registra também no Supabase (falha silenciosa)
    supabase_logger.log_prediction(
        features=wine.model_dump(),
        predicted_quality=quality,
        quality_label=CLASS_LABELS.get(quality, str(quality)),
        probabilities=probabilities,
        elapsed_ms=elapsed_ms,
    )

    return PredictionResponse(
        quality=quality,
        quality_label=CLASS_LABELS.get(quality, str(quality)),
        probabilities=probabilities,
        predicted_score=predicted_score,
        elapsed_ms=elapsed_ms,
    )


@app.get("/simulations", tags=["history"])
def list_simulations(limit: int = 50, db: Session = Depends(get_db)):
    """Retorna as últimas simulações registradas."""
    rows = (
        db.query(SimulationLog)
        .order_by(SimulationLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "predicted_quality": r.predicted_quality,
            "predicted_score": r.predicted_score,
            "quality_label": CLASS_LABELS.get(r.predicted_quality, ""),
            "elapsed_ms": r.elapsed_ms,
            "input_data": r.input_data,
        }
        for r in rows
    ]
