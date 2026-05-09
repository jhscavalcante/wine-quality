"""
Módulo de pré-processamento via DuckDB.

Lê o CSV bruto do vinho (padrão UCI com separador ';') e gera um
arquivo Parquet processado com as seguintes transformações:
  - Seleciona as 11 features químicas + coluna 'type' (red/white)
  - quality_raw   : nota original (inteiro 3-9)
  - quality_binary: label de classificação (1 = Good se quality >= 7, else 0)
"""
from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]  # raiz do projeto
RAW_PATH = ROOT / "data" / "raw" / "wine_quality.csv"          # CSV de entrada
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_processed.parquet"  # saída


def preprocess(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> None:
    """Lê o CSV bruto e escreve o Parquet processado usando DuckDB.

    O DuckDB é usado por ser extremamente rápido para transformações
    analíticas e suportar escrita direta em Parquet sem dependência de Spark.
    """
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Conexão em memória — não persiste arquivo .duckdb
    con = duckdb.connect()
    # Lê o CSV automaticamente inferindo separador e tipos
    con.execute(
        """
        CREATE OR REPLACE TABLE raw AS
        SELECT *
        FROM read_csv_auto(?)
        """,
        [str(raw_path)],
    )

    # Engenharia de features e exportação direta para Parquet via SQL
    con.execute(
        """
        COPY (
            WITH engineered AS (
                SELECT
                    fixed_acidity,
                    volatile_acidity,
                    citric_acid,
                    residual_sugar,
                    chlorides,
                    free_sulfur_dioxide,
                    total_sulfur_dioxide,
                    density,
                    ph,
                    sulphates,
                    alcohol,
                    type,
                    CAST(quality AS INTEGER) AS quality_raw,        -- nota numérica
                    CASE WHEN quality >= 7 THEN 1 ELSE 0 END AS quality_binary  -- label binário
                FROM raw
            )
            SELECT * FROM engineered
        ) TO ?
        (FORMAT PARQUET)
        """,
        [str(processed_path)],
    )
    con.close()

    print(f"[preprocessing] Dataset processado salvo em: {processed_path}")
    print("[preprocessing] Targets: quality_raw (3..9) e quality_binary (1=good>=7, 0=not_good<7)")


if __name__ == "__main__":
    preprocess()
