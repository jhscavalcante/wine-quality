FROM python:3.12-slim

WORKDIR /app

# System deps (curl for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (API + Streamlit UI)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files to app directory
COPY database.py   ./database.py
COPY models.py     ./models.py
COPY main.py       ./main.py
COPY streamlit_ui.py ./streamlit_ui.py
COPY supabase_logger.py ./supabase_logger.py

# Copy trained models
COPY models/ ./models/
COPY reports/ ./reports/

# Expor apenas Streamlit para acesso externo
EXPOSE 8501

# Start Streamlit UI directly (no FastAPI started by default)
# Note: the API will not be running inside this container when using this CMD.
# If you need the API too, run it separately or revert to starting `main:app`.
CMD ["python", "-m", "streamlit", "run", "streamlit_ui.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
