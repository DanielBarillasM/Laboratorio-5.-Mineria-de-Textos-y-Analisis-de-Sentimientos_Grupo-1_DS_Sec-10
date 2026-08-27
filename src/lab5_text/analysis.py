"""Validación, variables exploratorias y frecuencias de n-gramas."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessing import clean_for_classification, surface_markers


REQUIRED_COLUMNS = ("id", "keyword", "location", "text", "target")


def validate_dataset(frame: pd.DataFrame) -> None:
    """Valida el esquema mínimo y las etiquetas del conjunto de entrenamiento."""

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    if frame["id"].duplicated().any():
        raise ValueError("La columna id contiene duplicados")
    labels = set(frame["target"].dropna().unique())
    if labels != {0, 1}:
        raise ValueError(f"target debe contener exactamente 0 y 1; recibido: {labels}")
    if frame["text"].isna().any():
        raise ValueError("Hay tweets sin texto")


def add_eda_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Añade texto limpio, longitudes y marcadores sociales."""

    result = frame.copy()
    result["clean_text"] = result["text"].map(clean_for_classification)
    result["char_count"] = result["text"].str.len()
    result["word_count"] = result["text"].str.split().str.len()
    result["clean_word_count"] = result["clean_text"].str.split().str.len()
    result["unique_word_ratio"] = result["text"].map(
        lambda value: len(set(value.lower().split())) / max(1, len(value.split()))
    )
    markers = pd.DataFrame(result["text"].map(surface_markers).tolist(), index=result.index)
    result = pd.concat([result, markers], axis=1)
    result["keyword_present"] = result["keyword"].notna().astype(int)
    result["location_present"] = result["location"].notna().astype(int)
    return result


def ngram_frequencies(
    texts: Iterable[str],
    *,
    n: int,
    min_df: int = 2,
    top_k: int | None = 40,
) -> pd.DataFrame:
    """Calcula frecuencia y probabilidad empírica de n-gramas."""

    vectorizer = CountVectorizer(ngram_range=(n, n), min_df=min_df, lowercase=False)
    matrix = vectorizer.fit_transform(list(texts))
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    tokens = np.asarray(vectorizer.get_feature_names_out())
    order = np.argsort(-frequencies)
    result = pd.DataFrame({"ngram": tokens[order], "frecuencia": frequencies[order]})
    total = int(frequencies.sum())
    result["probabilidad"] = result["frecuencia"] / max(total, 1)
    result["n"] = n
    return result.head(top_k).reset_index(drop=True) if top_k else result.reset_index(drop=True)


def discriminative_unigrams(class_zero: pd.DataFrame, class_one: pd.DataFrame) -> pd.DataFrame:
    """Calcula un log-cociente suavizado para detectar términos distintivos."""

    left = class_zero[["ngram", "frecuencia"]].rename(columns={"frecuencia": "freq_0"})
    right = class_one[["ngram", "frecuencia"]].rename(columns={"frecuencia": "freq_1"})
    merged = left.merge(right, on="ngram", how="outer").fillna(0)
    total_0 = merged["freq_0"].sum()
    total_1 = merged["freq_1"].sum()
    vocabulary = len(merged)
    alpha = 0.5
    merged["log2_ratio_desastre"] = np.log2(
        ((merged["freq_1"] + alpha) / (total_1 + alpha * vocabulary))
        / ((merged["freq_0"] + alpha) / (total_0 + alpha * vocabulary))
    )
    merged["frecuencia_total"] = merged["freq_0"] + merged["freq_1"]
    return merged.sort_values("log2_ratio_desastre", ascending=False).reset_index(drop=True)
