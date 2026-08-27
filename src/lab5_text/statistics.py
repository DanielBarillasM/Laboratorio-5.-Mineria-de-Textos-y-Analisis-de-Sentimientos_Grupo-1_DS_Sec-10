"""Contraste no paramétrico entre clases para las medidas de sentimiento.

La pregunta que se responde es si los tweets de desastres reales son más
negativos que el resto. Como ``compound`` y ``negativity`` no son normales
—``negativity`` acumula masa en cero— se evita la prueba t y se usa:

* **Mann–Whitney U**, que compara distribuciones sin suponer normalidad.
* **Delta de Cliff**, tamaño de efecto derivado del propio estadístico U,
  interpretado con los cortes de Romano et al. (2006).
* **Intervalo bootstrap percentil al 95%** para la diferencia de medias y de
  medianas, que cuantifica la incertidumbre del efecto en la escala original.

Ninguna de estas herramientas establece causalidad: describen una asociación
observacional entre la etiqueta del tweet y su polaridad estimada.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


RANDOM_STATE = 42
BOOTSTRAP_RESAMPLES = 10_000
CHUNK = 500


@dataclass(frozen=True)
class ComparisonResult:
    """Resumen completo del contraste entre dos grupos independientes."""

    metrica: str
    n_grupo_1: int
    n_grupo_0: int
    media_grupo_1: float
    media_grupo_0: float
    mediana_grupo_1: float
    mediana_grupo_0: float
    diferencia_medias: float
    diferencia_medianas: float
    u_statistic: float
    p_valor_bilateral: float
    p_valor_unilateral_mayor: float
    cliffs_delta: float
    magnitud_efecto: str
    ic95_dif_medias_inferior: float
    ic95_dif_medias_superior: float
    ic95_dif_medianas_inferior: float
    ic95_dif_medianas_superior: float

    def to_dict(self) -> dict:
        return asdict(self)


def cliffs_delta_magnitude(delta: float) -> str:
    """Clasifica el tamaño de efecto con los cortes de Romano et al. (2006)."""

    magnitude = abs(delta)
    if magnitude < 0.147:
        return "insignificante"
    if magnitude < 0.330:
        return "pequeño"
    if magnitude < 0.474:
        return "mediano"
    return "grande"


def cliffs_delta(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Delta de Cliff de ``group_a`` frente a ``group_b``, en [-1, 1].

    Se obtiene del estadístico U de Mann–Whitney, que ya reparte los empates a
    la mitad: ``delta = 2U / (n_a * n_b) - 1``.
    """

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    u_statistic = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u_statistic / (a.size * b.size) - 1.0)


def bootstrap_difference_ci(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    statistic: str = "mean",
    resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = RANDOM_STATE,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Intervalo percentil bootstrap para ``stat(a) - stat(b)``."""

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    reducer = {"mean": np.mean, "median": np.median}[statistic]
    generator = np.random.default_rng(random_state)
    differences = np.empty(resamples, dtype=float)
    done = 0
    while done < resamples:
        size = min(CHUNK, resamples - done)
        sample_a = a[generator.integers(0, a.size, size=(size, a.size))]
        sample_b = b[generator.integers(0, b.size, size=(size, b.size))]
        differences[done : done + size] = reducer(sample_a, axis=1) - reducer(sample_b, axis=1)
        done += size
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def compare_groups(
    group_1: np.ndarray,
    group_0: np.ndarray,
    *,
    metric: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = RANDOM_STATE,
) -> ComparisonResult:
    """Compara la clase de desastre (grupo 1) contra el resto (grupo 0)."""

    a = np.asarray(group_1, dtype=float)
    b = np.asarray(group_0, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("Ambos grupos deben tener al menos una observación")

    two_sided = stats.mannwhitneyu(a, b, alternative="two-sided")
    greater = stats.mannwhitneyu(a, b, alternative="greater")
    delta = float(2.0 * two_sided.statistic / (a.size * b.size) - 1.0)
    mean_low, mean_high = bootstrap_difference_ci(
        a, b, statistic="mean", resamples=resamples, random_state=random_state
    )
    median_low, median_high = bootstrap_difference_ci(
        a, b, statistic="median", resamples=resamples, random_state=random_state
    )
    return ComparisonResult(
        metrica=metric,
        n_grupo_1=int(a.size),
        n_grupo_0=int(b.size),
        media_grupo_1=float(a.mean()),
        media_grupo_0=float(b.mean()),
        mediana_grupo_1=float(np.median(a)),
        mediana_grupo_0=float(np.median(b)),
        diferencia_medias=float(a.mean() - b.mean()),
        diferencia_medianas=float(np.median(a) - np.median(b)),
        u_statistic=float(two_sided.statistic),
        p_valor_bilateral=float(two_sided.pvalue),
        p_valor_unilateral_mayor=float(greater.pvalue),
        cliffs_delta=delta,
        magnitud_efecto=cliffs_delta_magnitude(delta),
        ic95_dif_medias_inferior=mean_low,
        ic95_dif_medias_superior=mean_high,
        ic95_dif_medianas_inferior=median_low,
        ic95_dif_medianas_superior=median_high,
    )
