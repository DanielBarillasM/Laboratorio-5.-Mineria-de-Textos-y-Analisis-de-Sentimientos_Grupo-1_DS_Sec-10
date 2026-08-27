import pandas as pd
import pytest

from lab5_text.analysis import add_eda_features, ngram_frequencies, validate_dataset


def test_validate_official_schema() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "keyword": ["fire", None],
            "location": [None, "Guatemala"],
            "text": ["Forest fire", "Movie disaster"],
            "target": [1, 0],
        }
    )
    validate_dataset(frame)


def test_validate_rejects_invalid_target() -> None:
    frame = pd.DataFrame(
        {"id": [1], "keyword": [None], "location": [None], "text": ["x"], "target": [2]}
    )
    with pytest.raises(ValueError):
        validate_dataset(frame)


def test_ngram_probabilities_are_normalized() -> None:
    result = ngram_frequencies(pd.Series(["fire near city", "city fire"]), n=1, top_k=None)

    # "near" pertenece a las stopwords del vectorizador y se excluye.
    assert result["frecuencia"].sum() == 4
    assert result["probabilidad"].sum() == pytest.approx(1.0)


def test_eda_features_include_clean_text_and_lengths() -> None:
    frame = pd.DataFrame(
        {"id": [1], "keyword": ["fire"], "location": [None], "text": ["No #fire!"], "target": [1]}
    )
    enriched = add_eda_features(frame)

    assert enriched.loc[0, "clean_text"] == "no"
    assert enriched.loc[0, "char_count"] == len("No #fire!")
    assert enriched.loc[0, "word_count"] == 2
