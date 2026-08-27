"""Contrato del contraste estadístico entre clases."""

import numpy as np
import pytest

from lab5_text.statistics import (
    bootstrap_difference_ci,
    cliffs_delta,
    cliffs_delta_magnitude,
    compare_groups,
)


def test_cliffs_delta_is_zero_for_identical_distributions() -> None:
    values = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

    assert cliffs_delta(values, values) == pytest.approx(0.0)


def test_cliffs_delta_reaches_its_extremes_when_groups_are_disjoint() -> None:
    low = np.array([0.0, 0.1, 0.2])
    high = np.array([0.8, 0.9, 1.0])

    assert cliffs_delta(high, low) == pytest.approx(1.0)
    assert cliffs_delta(low, high) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (0.0, "insignificante"),
        (0.10, "insignificante"),
        (0.20, "pequeño"),
        (-0.20, "pequeño"),
        (0.40, "mediano"),
        (0.60, "grande"),
        (-0.95, "grande"),
    ],
)
def test_cliffs_delta_magnitude_uses_romano_thresholds(delta: float, expected: str) -> None:
    assert cliffs_delta_magnitude(delta) == expected


def test_bootstrap_interval_is_deterministic_and_brackets_the_difference() -> None:
    generator = np.random.default_rng(7)
    group_a = generator.normal(1.0, 0.5, 400)
    group_b = generator.normal(0.0, 0.5, 400)

    low, high = bootstrap_difference_ci(group_a, group_b, resamples=1_000, random_state=42)
    repeated = bootstrap_difference_ci(group_a, group_b, resamples=1_000, random_state=42)

    assert (low, high) == repeated
    assert low < high
    assert low <= float(group_a.mean() - group_b.mean()) <= high


def test_bootstrap_supports_the_median_statistic() -> None:
    generator = np.random.default_rng(11)
    group_a = generator.normal(2.0, 1.0, 200)
    group_b = generator.normal(0.0, 1.0, 200)

    low, high = bootstrap_difference_ci(
        group_a, group_b, statistic="median", resamples=500, random_state=42
    )

    assert low < high
    assert low > 0.0


def test_compare_groups_returns_the_full_documented_contract() -> None:
    generator = np.random.default_rng(3)
    disaster = generator.uniform(0.2, 1.0, 300)
    other = generator.uniform(0.0, 0.8, 400)

    result = compare_groups(disaster, other, metric="negativity", resamples=500)
    payload = result.to_dict()

    assert payload["metrica"] == "negativity"
    assert payload["n_grupo_1"] == 300 and payload["n_grupo_0"] == 400
    assert 0.0 <= payload["p_valor_bilateral"] <= 1.0
    assert 0.0 <= payload["p_valor_unilateral_mayor"] <= 1.0
    assert -1.0 <= payload["cliffs_delta"] <= 1.0
    assert payload["magnitud_efecto"] in {"insignificante", "pequeño", "mediano", "grande"}
    assert payload["ic95_dif_medias_inferior"] <= payload["ic95_dif_medias_superior"]
    assert payload["diferencia_medias"] == pytest.approx(
        payload["media_grupo_1"] - payload["media_grupo_0"]
    )


def test_compare_groups_detects_the_direction_of_the_effect() -> None:
    generator = np.random.default_rng(5)
    higher = generator.normal(0.6, 0.2, 250)
    lower = generator.normal(0.2, 0.2, 250)

    result = compare_groups(higher, lower, metric="demo", resamples=500)

    assert result.cliffs_delta > 0
    assert result.diferencia_medias > 0
    assert result.p_valor_unilateral_mayor < 0.01


def test_compare_groups_rejects_an_empty_group() -> None:
    with pytest.raises(ValueError):
        compare_groups(np.array([]), np.array([0.1, 0.2]), metric="demo", resamples=100)
