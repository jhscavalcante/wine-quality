"""
Módulo de particionamento do dataset processado em treino/val/teste.

Estrátégia de split estratificado (pela label binária) para garantir
proportions semelhantes da classe Good/Not Good em cada partição:
  - Treino : 60% dos dados
  - Validação: 20% (usado para ajustar o limiar de binarização)
  - Teste  : 20% (avaliação final isolada)

Os splits são salvos em Parquet com as colunas de feature + quality_raw + quality_binary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_processed.parquet"  # entrada
SPLITS_DIR = ROOT / "data" / "processed" / "splits"                       # saída

TARGET_BINARY = "quality_binary"  # coluna usada para estratificar o split
TARGET_RAW = "quality_raw"        # coluna de nota numérica


def prepare(
    processed_path: Path = PROCESSED_PATH,
    splits_dir: Path = SPLITS_DIR,
) -> None:
    df = pd.read_parquet(processed_path)
    print(f"[prepare] Dados carregados: {df.shape}")

    if TARGET_RAW not in df.columns or TARGET_BINARY not in df.columns:
        raise ValueError(
            f"Colunas esperadas ausentes. Esperado: {TARGET_RAW} e {TARGET_BINARY}."
        )

    X = df.drop(columns=[TARGET_RAW, TARGET_BINARY])
    y_binary = df[TARGET_BINARY].astype(int)
    y_raw = df[TARGET_RAW].astype(float)

    # --- Primeiro split: separa 20% para teste (isolado) ---
    # stratify=y_binary garante proporção similar de Good/Not Good no teste
    X_temp, X_test, y_bin_temp, y_bin_test, y_raw_temp, y_raw_test = train_test_split(
        X,
        y_binary,
        y_raw,
        test_size=0.20,
        random_state=42,
        stratify=y_binary,
    )

    # --- Segundo split: 25% do restante vira validação (= 20% do total) ---
    # Resultado final: 60% treino | 20% val | 20% teste
    X_train, X_val, y_bin_train, y_bin_val, y_raw_train, y_raw_val = train_test_split(
        X_temp,
        y_bin_temp,
        y_raw_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_bin_temp,
    )

    print("[prepare] Split 60/20/20 concluído")
    print(f"  train={len(X_train)} | val={len(X_val)} | test={len(X_test)} | total={len(X)}")

    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, X_split, y_bin_split, y_raw_split in [
        ("train", X_train, y_bin_train, y_raw_train),
        ("val", X_val, y_bin_val, y_raw_val),
        ("test", X_test, y_bin_test, y_raw_test),
    ]:
        out_df = X_split.copy()
        out_df[TARGET_BINARY] = y_bin_split.values
        out_df[TARGET_RAW] = y_raw_split.values
        out_path = splits_dir / f"{split_name}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"[prepare] {split_name:>5} -> {out_path.name} ({len(out_df)} linhas)")

    print("[prepare] ✅ Splits salvos com targets raw/binary")


if __name__ == "__main__":
    prepare()
