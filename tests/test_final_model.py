"""Contrato del reentrenamiento, la selección final y la función predictiva."""

import numpy as np
import pandas as pd
import pytest

from lab5_text.final_model import (
    MODEL_A,
    MODEL_B,
    build_final_pipeline,
    classify_raw_tweet,
    compare_negativity,
    select_final_model,
    stratified_split,
)


@pytest.fixture(scope="module")
def toy_frame() -> pd.DataFrame:
    """Corpus sintético pequeño, balanceado y separable, para pruebas rápidas."""

    disaster = [
        "wildfire evacuation emergency crews",
        "earthquake collapse buildings rescue",
        "flood damage homes destroyed",
        "hurricane landfall storm surge",
        "explosion downtown casualties reported",
    ]
    ordinary = [
        "movie tonight popcorn friends",
        "coffee morning routine music",
        "football match great goal",
        "birthday party cake balloons",
        "shopping mall new shoes",
    ]
    texts = (disaster * 12) + (ordinary * 12)
    targets = ([1] * 60) + ([0] * 60)
    generator = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "clean_text": texts,
            "negativity": generator.uniform(0.0, 1.0, len(texts)),
            "target": targets,
        }
    )


def test_pipeline_without_negativity_uses_only_the_text_branch() -> None:
    transformers = build_final_pipeline(use_negativity=False).named_steps["features"].transformers

    assert [name for name, _, _ in transformers] == ["texto"]


def test_pipeline_with_negativity_adds_a_scaled_numeric_branch() -> None:
    transformers = build_final_pipeline(use_negativity=True).named_steps["features"].transformers
    names = [name for name, _, _ in transformers]

    assert names == ["texto", "negatividad"]
    assert transformers[1][2] == ["negativity"]


def test_split_is_disjoint_complete_and_stratified() -> None:
    target = pd.Series([1] * 3271 + [0] * 4342)
    train_idx, test_idx = stratified_split(target)

    assert len(train_idx) == 6090
    assert len(test_idx) == 1523
    assert set(train_idx).isdisjoint(set(test_idx))
    assert len(set(train_idx) | set(test_idx)) == len(target)
    train_rate = target.iloc[train_idx].mean()
    test_rate = target.iloc[test_idx].mean()
    assert train_rate == pytest.approx(test_rate, abs=0.01)


def test_split_is_reproducible_across_calls() -> None:
    target = pd.Series([1, 0] * 200)
    first_train, first_test = stratified_split(target)
    second_train, second_test = stratified_split(target)

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_test, second_test)


def test_model_with_the_numeric_variable_trains_and_predicts(toy_frame: pd.DataFrame) -> None:
    model = build_final_pipeline(use_negativity=True)
    model.fit(toy_frame[["clean_text", "negativity"]], toy_frame["target"])
    probability = model.predict_proba(toy_frame[["clean_text", "negativity"]])[:, 1]

    assert probability.shape == (len(toy_frame),)
    assert ((probability >= 0.0) & (probability <= 1.0)).all()


def test_compare_negativity_evaluates_both_models_on_the_same_split(toy_frame: pd.DataFrame) -> None:
    comparison = compare_negativity(toy_frame)

    assert list(comparison.metrics["modelo"]) == [MODEL_A, MODEL_B]
    assert list(comparison.metrics["usa_negatividad"]) == [False, True]
    assert set(comparison.train_indices).isdisjoint(set(comparison.test_indices))
    assert len(comparison.differences) == 10
    assert set(comparison.differences["sentido"]) <= {"mejora", "empeora", "sin cambio"}
    for row in comparison.metrics.itertuples():
        assert row.tn + row.fp + row.fn + row.tp == len(comparison.test_indices)


def test_compare_negativity_requires_the_expected_columns() -> None:
    with pytest.raises(ValueError, match="columnas requeridas"):
        compare_negativity(pd.DataFrame({"clean_text": ["a"], "target": [1]}))


def test_selection_never_lets_roc_auc_override_a_clearly_better_f1() -> None:
    """Regresión: un F1 muy inferior no debe ganar por un ROC-AUC marginal."""

    metrics = pd.DataFrame(
        [
            {"modelo": MODEL_A, "f1": 0.780938, "roc_auc": 0.866635},
            {"modelo": MODEL_B, "f1": 0.763804, "roc_auc": 0.867043},
        ]
    )
    winner, reason = select_final_model(metrics)

    assert winner == MODEL_A
    assert "sin necesidad de desempate" in reason


def test_selection_uses_roc_auc_only_on_a_practical_tie() -> None:
    metrics = pd.DataFrame(
        [
            {"modelo": MODEL_A, "f1": 0.7810, "roc_auc": 0.8600},
            {"modelo": MODEL_B, "f1": 0.7801, "roc_auc": 0.8700},
        ]
    )
    winner, reason = select_final_model(metrics)

    assert winner == MODEL_B
    assert "ROC-AUC" in reason


def test_selection_requires_more_than_one_candidate() -> None:
    with pytest.raises(ValueError):
        select_final_model(pd.DataFrame([{"modelo": MODEL_A, "f1": 0.8, "roc_auc": 0.9}]))


def test_final_function_accepts_raw_text_and_returns_every_required_field(
    toy_frame: pd.DataFrame,
) -> None:
    model = build_final_pipeline(use_negativity=True)
    model.fit(toy_frame[["clean_text", "negativity"]], toy_frame["target"])

    result = classify_raw_tweet(model, "BREAKING: wildfire evacuation @city https://t.co/x #fire")

    assert set(result) == {
        "texto",
        "texto_limpio",
        "clase",
        "probabilidad_desastre",
        "sentimiento",
        "compound",
        "negatividad",
    }
    assert result["clase"] in {"Desastre real", "No desastre"}
    assert 0.0 <= result["probabilidad_desastre"] <= 1.0
    assert result["sentimiento"] in {"negativo", "neutral", "positivo"}
    assert -1.0 <= result["compound"] <= 1.0
    assert 0.0 <= result["negatividad"] <= 1.0
    # La limpieza del entrenamiento se aplica dentro de la función.
    assert "https" not in result["texto_limpio"] and "@city" not in result["texto_limpio"]


def test_final_function_works_on_a_text_only_model(toy_frame: pd.DataFrame) -> None:
    model = build_final_pipeline(use_negativity=False)
    model.fit(toy_frame[["clean_text", "negativity"]], toy_frame["target"])

    result = classify_raw_tweet(model, "Earthquake collapse downtown, rescue crews deployed")

    assert result["clase"] == "Desastre real"
    assert result["texto_limpio"]


def test_final_function_tolerates_empty_and_non_string_input(toy_frame: pd.DataFrame) -> None:
    model = build_final_pipeline(use_negativity=True)
    model.fit(toy_frame[["clean_text", "negativity"]], toy_frame["target"])

    for value in ("", None, 12345):
        result = classify_raw_tweet(model, value)
        assert result["texto_limpio"] == ""
        assert result["sentimiento"] == "neutral"
        assert result["negatividad"] == 0.0
