from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "wine_quality.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_processed.parquet"


def preprocess(raw_path: Path = RAW_PATH, processed_path: Path = PROCESSED_PATH) -> None:
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(
        """
        CREATE OR REPLACE TABLE raw AS
        SELECT *
        FROM read_csv_auto(?)
        """,
        [str(raw_path)],
    )

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
                    CAST(quality AS INTEGER) AS quality_raw,
                    CASE WHEN quality >= 7 THEN 1 ELSE 0 END AS quality_binary
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
