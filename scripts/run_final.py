"""Flujo reproducible de la entrega final del Laboratorio 5.

Regenera, en un solo paso y con semilla fija, todos los resultados que citan el
notebook y el informe: tablas de sentimiento, top 10 por polaridad, contraste
estadístico entre clases, comparación del modelo sin y con la variable de
negatividad, métricas finales, matrices de confusión, análisis de errores,
figuras, el dataset procesado, el modelo definitivo y la evidencia de rúbrica.

Uso
---
    python3 scripts/run_final.py
"""

from __future__ import annotations

from collections import Counter
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.insert(0, str(ROOT / "src"))

from lab5_text.analysis import add_eda_features, validate_dataset  # noqa: E402
from lab5_text.final_model import (  # noqa: E402
    LOGISTIC_PARAMS,
    MODEL_B,
    TFIDF_PARAMS,
    NegativityComparison,
    classify_raw_tweet,
    compare_negativity,
    select_final_model,
)
from lab5_text.modeling import evaluate_models  # noqa: E402
from lab5_text.preprocessing import STOPWORDS, surface_markers  # noqa: E402
from lab5_text.sentiment import (  # noqa: E402
    NEGATIVE_THRESHOLD,
    POSITIVE_THRESHOLD,
    add_sentiment,
    extreme_tweets,
    sentiment_summary,
)
from lab5_text.statistics import compare_groups  # noqa: E402


RAW = ROOT / "data" / "raw" / "train.csv"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "models"

RANDOM_STATE = 42
NEGATION_RE = re.compile(r"\b(no|not|never|none|nor|neither|cannot|n't)\b", re.IGNORECASE)
BLUE = "#2563eb"
RED = "#dc2626"
NAVY = "#102a43"
TEAL = "#0f9d91"
ORANGE = "#f59e0b"
GREY = "#94a3b8"
CATEGORY = {0: "No desastre", 1: "Desastre real"}
#: Umbral de apariciones para que una palabra vacía entre en la auditoría.
MIN_STOPWORD_OCCURRENCES = 25
#: Palabras con carga temática que la lista Glasgow de scikit-learn descarta.
CONTENT_STOPWORDS = {
    "fire", "cry", "call", "serious", "system", "front", "back", "side", "top",
    "bottom", "full", "empty", "move", "show", "find", "part", "well", "bill",
    "detail", "amount", "interest", "describe", "fill", "thick", "thin",
    "hundred", "mill", "sincere",
}
LABEL_COLORS = {"negativo": RED, "neutral": GREY, "positivo": TEAL}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7fafc",
            "axes.facecolor": "#ffffff",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
        }
    )



# --------------------------------------------------------------------------- #
# Etapa 0 · Auditoría de la lista de palabras vacías
# --------------------------------------------------------------------------- #
def stage_stopword_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Documenta qué palabras de contenido elimina la lista de scikit-learn.

    ``ENGLISH_STOP_WORDS`` es la lista Glasgow, que además de palabras
    funcionales incluye sustantivos y verbos con carga temática, entre ellos
    ``fire``. La tabla resultante cuantifica esa pérdida por clase y sirve de
    respaldo para la sección de limitaciones del informe: se conoce, se mide y
    se declara, en vez de quedar como un efecto silencioso del preprocesamiento.
    """

    counters = {0: Counter(), 1: Counter()}
    for text, target in zip(frame["text"], frame["target"]):
        counters[int(target)].update(set(str(text).lower().split()))

    rows = []
    for word in STOPWORDS:
        n_1, n_0 = counters[1][word], counters[0][word]
        if n_1 + n_0 < MIN_STOPWORD_OCCURRENCES:
            continue
        odds = ((n_1 + 0.5) / max(1, counters[1].total())) / (
            (n_0 + 0.5) / max(1, counters[0].total())
        )
        rows.append(
            {
                "palabra_eliminada": word,
                "tweets_no_desastre": int(n_0),
                "tweets_desastre": int(n_1),
                "log2_razon_desastre": float(np.log2(odds)),
                "es_palabra_de_contenido": word in CONTENT_STOPWORDS,
            }
        )
    audit = pd.DataFrame(rows).sort_values("log2_razon_desastre", ascending=False)
    audit.to_csv(TABLES / "auditoria_stopwords.csv", index=False)
    return audit

# --------------------------------------------------------------------------- #
# Etapa 1 · Análisis final de sentimiento
# --------------------------------------------------------------------------- #
def stage_sentiment(frame: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta cada tweet, resume por clase y grafica la distribución."""

    sentiment = add_sentiment(frame, analyzer=SentimentIntensityAnalyzer())
    sentiment["categoria"] = sentiment["target"].map(CATEGORY)

    summary = sentiment_summary(sentiment)
    summary.to_csv(TABLES / "sentimiento_distribucion.csv", index=False)

    global_summary = (
        sentiment["sentiment_label"]
        .value_counts()
        .rename_axis("sentiment_label")
        .reset_index(name="tweets")
    )
    global_summary["porcentaje"] = 100 * global_summary["tweets"] / len(sentiment)
    global_summary.to_csv(TABLES / "sentimiento_distribucion_global.csv", index=False)

    measures = [
        "sentiment_compound",
        "negativity",
        "sentiment_negative",
        "sentiment_positive",
        "sentiment_neutral",
        "pos_word_count",
        "neg_word_count",
        "neu_word_count",
        "token_count",
    ]
    statistics = sentiment.groupby("target")[measures].agg(["count", "mean", "median", "std"])
    statistics.columns = [f"{left}_{right}" for left, right in statistics.columns]
    statistics = statistics.reset_index()
    statistics["categoria"] = statistics["target"].map(CATEGORY)
    statistics.to_csv(TABLES / "sentimiento_estadisticos_por_clase.csv", index=False)

    thresholds = pd.DataFrame(
        [
            {
                "etiqueta": "positivo",
                "regla": f"compound >= {POSITIVE_THRESHOLD}",
                "fuente": "Hutto y Gilbert (2014)",
            },
            {
                "etiqueta": "negativo",
                "regla": f"compound <= {NEGATIVE_THRESHOLD}",
                "fuente": "Hutto y Gilbert (2014)",
            },
            {
                "etiqueta": "neutral",
                "regla": f"{NEGATIVE_THRESHOLD} < compound < {POSITIVE_THRESHOLD}",
                "fuente": "Hutto y Gilbert (2014)",
            },
        ]
    )
    thresholds.to_csv(TABLES / "sentimiento_umbrales.csv", index=False)

    words = (
        sentiment.groupby("categoria")[["pos_word_count", "neg_word_count", "neu_word_count"]]
        .mean()
        .reset_index()
        .melt(id_vars="categoria", var_name="tipo", value_name="promedio")
    )
    words["tipo"] = words["tipo"].map(
        {
            "pos_word_count": "Positivas",
            "neg_word_count": "Negativas",
            "neu_word_count": "Neutrales",
        }
    )
    words.to_csv(TABLES / "sentimiento_conteo_palabras.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    sns.barplot(
        data=summary,
        x="sentiment_label",
        y="porcentaje",
        hue="categoria",
        order=["negativo", "neutral", "positivo"],
        palette=[BLUE, RED],
        ax=axes[0, 0],
    )
    axes[0, 0].set(title="Sentimiento VADER por clase", xlabel="", ylabel="Tweets de la clase (%)")
    axes[0, 0].legend(title="")
    for container in axes[0, 0].containers:
        axes[0, 0].bar_label(container, fmt="%.1f", fontsize=8)

    sns.violinplot(
        data=sentiment,
        x="categoria",
        y="sentiment_compound",
        hue="categoria",
        palette=[BLUE, RED],
        legend=False,
        inner="quart",
        cut=0,
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(POSITIVE_THRESHOLD, color=TEAL, linestyle="--", linewidth=1)
    axes[0, 1].axhline(NEGATIVE_THRESHOLD, color=RED, linestyle="--", linewidth=1)
    axes[0, 1].set(title="Polaridad compuesta y umbrales", xlabel="", ylabel="compound [-1, 1]")

    sns.barplot(
        data=words, x="tipo", y="promedio", hue="categoria", palette=[BLUE, RED], ax=axes[1, 0]
    )
    axes[1, 0].set(
        title="Palabras por tweet según el lexicón", xlabel="", ylabel="Promedio de tokens"
    )
    axes[1, 0].legend(title="")
    for container in axes[1, 0].containers:
        axes[1, 0].bar_label(container, fmt="%.2f", fontsize=8)

    sns.ecdfplot(
        data=sentiment, x="negativity", hue="categoria", palette={"No desastre": BLUE, "Desastre real": RED}, ax=axes[1, 1]
    )
    axes[1, 1].set(
        title="Acumulada de la negatividad", xlabel="negativity = max(-compound, 0)", ylabel="Proporción"
    )
    fig.suptitle("Análisis final de sentimiento", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "sentimiento_final.png", dpi=180)
    plt.close(fig)
    return sentiment


# --------------------------------------------------------------------------- #
# Etapa 2 · Top 10 positivos y negativos
# --------------------------------------------------------------------------- #
def stage_top_tweets(sentiment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrae y guarda los diez tweets más positivos y los diez más negativos."""

    positives, negatives = extreme_tweets(sentiment, k=10)
    positives.to_csv(TABLES / "top10_positivos.csv", index=False)
    negatives.to_csv(TABLES / "top10_negativos.csv", index=False)

    composition = pd.concat(
        [
            positives.assign(extremo="Top 10 positivos"),
            negatives.assign(extremo="Top 10 negativos"),
        ]
    )
    breakdown = (
        composition.groupby(["extremo", "categoria"]).size().rename("tweets").reset_index()
    )
    breakdown.to_csv(TABLES / "top10_composicion.csv", index=False)

    patterns = composition.assign(
        marcas=composition["text"].map(surface_markers).tolist(),
    )
    marker_frame = pd.DataFrame(list(patterns["marcas"]), index=patterns.index)
    patterns = pd.concat([patterns.drop(columns=["marcas"]), marker_frame], axis=1)
    patterns["mayusculas_pct"] = patterns["text"].map(
        lambda value: 100 * sum(char.isupper() for char in value) / max(1, len(value))
    )
    patterns["tokens_repetidos_max"] = patterns["text"].map(
        lambda value: max(Counter(value.lower().split()).values(), default=0)
    )
    patterns["contiene_negacion"] = patterns["text"].map(
        lambda value: int(bool(NEGATION_RE.search(value)))
    )
    patterns[
        [
            "extremo",
            "id",
            "categoria",
            "sentiment_compound",
            "negativity",
            "pos_word_count",
            "neg_word_count",
            "url_count",
            "mention_count",
            "hashtag_count",
            "exclamation_count",
            "question_count",
            "mayusculas_pct",
            "tokens_repetidos_max",
            "contiene_negacion",
        ]
    ].to_csv(TABLES / "top10_patrones.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(
        data=breakdown, x="extremo", y="tweets", hue="categoria", palette=[BLUE, RED], ax=ax
    )
    ax.set(
        title="Composición de los extremos de polaridad",
        xlabel="",
        ylabel="Tweets (de 10)",
        ylim=(0, 10.5),
    )
    ax.legend(title="")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    fig.tight_layout()
    fig.savefig(FIGURES / "top10_composicion.png", dpi=180)
    plt.close(fig)
    return positives, negatives


# --------------------------------------------------------------------------- #
# Etapa 3 · Contraste estadístico entre clases
# --------------------------------------------------------------------------- #
def stage_statistics(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Prueba si la clase de desastre es más negativa que el resto."""

    rows = []
    for metric in ("negativity", "sentiment_negative", "sentiment_compound"):
        group_1 = sentiment.loc[sentiment["target"] == 1, metric].to_numpy()
        group_0 = sentiment.loc[sentiment["target"] == 0, metric].to_numpy()
        rows.append(compare_groups(group_1, group_0, metric=metric).to_dict())
    contrast = pd.DataFrame(rows)
    contrast.to_csv(TABLES / "contraste_estadistico.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    sns.kdeplot(
        data=sentiment,
        x="sentiment_compound",
        hue="categoria",
        palette={"No desastre": BLUE, "Desastre real": RED},
        fill=True,
        common_norm=False,
        alpha=0.35,
        ax=axes[0],
    )
    axes[0].set(title="Densidad de compound por clase", xlabel="compound", ylabel="Densidad")

    sns.boxplot(
        data=sentiment,
        x="categoria",
        y="negativity",
        hue="categoria",
        palette=[BLUE, RED],
        legend=False,
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set(title="Negatividad por clase", xlabel="", ylabel="negativity")

    positive_only = sentiment[sentiment["negativity"] > 0]
    sns.histplot(
        data=positive_only,
        x="negativity",
        hue="categoria",
        palette={"No desastre": BLUE, "Desastre real": RED},
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[2],
    )
    axes[2].set(
        title="Negatividad estrictamente positiva", xlabel="negativity > 0", ylabel="Densidad"
    )
    delta = contrast.loc[contrast["metrica"] == "negativity", "cliffs_delta"].iloc[0]
    magnitude = contrast.loc[contrast["metrica"] == "negativity", "magnitud_efecto"].iloc[0]
    fig.suptitle(
        f"¿Los desastres reales son más negativos?  ·  delta de Cliff = {delta:.3f} ({magnitude})",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "contraste_negatividad.png", dpi=180)
    plt.close(fig)
    return contrast


# --------------------------------------------------------------------------- #
# Etapa 4 · Definición y auditoría de la variable de negatividad
# --------------------------------------------------------------------------- #
def stage_negativity(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Documenta y valida el rango y el comportamiento de ``negativity``."""

    values = sentiment["negativity"]
    audit = pd.DataFrame(
        [
            {"indicador": "definicion", "valor": "negativity = max(-compound, 0)"},
            {"indicador": "rango_teorico", "valor": "[0, 1]"},
            {"indicador": "minimo_observado", "valor": float(values.min())},
            {"indicador": "maximo_observado", "valor": float(values.max())},
            {"indicador": "media", "valor": float(values.mean())},
            {"indicador": "mediana", "valor": float(values.median())},
            {"indicador": "desviacion", "valor": float(values.std())},
            {"indicador": "tweets_con_negatividad_cero", "valor": int((values == 0).sum())},
            {"indicador": "pct_negatividad_cero", "valor": float(100 * (values == 0).mean())},
            {
                "indicador": "correlacion_spearman_con_neg_vader",
                "valor": float(values.corr(sentiment["sentiment_negative"], method="spearman")),
            },
            {
                "indicador": "correlacion_spearman_con_target",
                "valor": float(values.corr(sentiment["target"].astype(float), method="spearman")),
            },
        ]
    )
    audit.to_csv(TABLES / "negatividad_definicion.csv", index=False)

    bins = pd.cut(values, bins=[-0.001, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    distribution = (
        sentiment.assign(rango=bins.astype(str))
        .groupby(["rango", "categoria"])
        .size()
        .rename("tweets")
        .reset_index()
    )
    distribution["porcentaje"] = 100 * distribution["tweets"] / distribution.groupby("categoria")[
        "tweets"
    ].transform("sum")
    distribution.to_csv(TABLES / "negatividad_por_rango.csv", index=False)
    return audit



# --------------------------------------------------------------------------- #
# Etapa 5 · Modelos candidatos sobre la partición común
# --------------------------------------------------------------------------- #
def stage_candidate_models(sentiment: pd.DataFrame) -> tuple[pd.DataFrame, dict, NegativityComparison]:
    """Evalúa los cuatro modelos base y el experimento A/B en la misma validación."""

    preliminary = evaluate_models(sentiment["clean_text"], sentiment["target"])
    preliminary.metrics.to_csv(TABLES / "metricas_modelos_preliminares.csv", index=False)
    preliminary.confusion.to_csv(TABLES / "matrices_confusion.csv", index=False)
    preliminary.roc_points.to_csv(TABLES / "curvas_roc.csv", index=False)

    comparison = compare_negativity(sentiment)
    comparison.metrics.to_csv(TABLES / "comparacion_negatividad_metricas.csv", index=False)
    comparison.differences.to_csv(TABLES / "comparacion_negatividad_diferencias.csv", index=False)

    if not np.array_equal(preliminary.test_indices, comparison.test_indices):
        raise RuntimeError("Las particiones de validación no coinciden entre experimentos")

    split = pd.DataFrame(
        [
            {
                "particion": "entrenamiento",
                "filas": len(comparison.train_indices),
                "desastre": int(sentiment.iloc[comparison.train_indices]["target"].sum()),
                "desastre_pct": 100 * sentiment.iloc[comparison.train_indices]["target"].mean(),
            },
            {
                "particion": "validacion",
                "filas": len(comparison.test_indices),
                "desastre": int(sentiment.iloc[comparison.test_indices]["target"].sum()),
                "desastre_pct": 100 * sentiment.iloc[comparison.test_indices]["target"].mean(),
            },
        ]
    )
    split.to_csv(TABLES / "particion_modelado.csv", index=False)

    columns = [
        "modelo",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f1_macro",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "n_train",
        "n_test",
    ]
    catalog = pd.concat(
        [
            preliminary.metrics.assign(familia="preliminar", usa_negatividad=False)[
                columns + ["familia", "usa_negatividad"]
            ],
            comparison.metrics.assign(familia="reentrenamiento")[
                columns + ["familia", "usa_negatividad"]
            ],
        ],
        ignore_index=True,
    ).sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)
    catalog.to_csv(TABLES / "metricas_todos_los_modelos.csv", index=False)

    fitted = dict(preliminary.fitted_models)
    fitted.update(comparison.fitted_models)

    plot_frame = comparison.differences[
        comparison.differences["metrica"].isin(
            ["accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"]
        )
    ].melt(
        id_vars="metrica",
        value_vars=["modelo_a_sin_negatividad", "modelo_b_con_negatividad"],
        var_name="modelo",
        value_name="valor",
    )
    plot_frame["modelo"] = plot_frame["modelo"].map(
        {"modelo_a_sin_negatividad": "A · sin negatividad", "modelo_b_con_negatividad": "B · con negatividad"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    sns.barplot(data=plot_frame, x="metrica", y="valor", hue="modelo", palette=[BLUE, ORANGE], ax=axes[0])
    axes[0].set(title="Modelo A frente a modelo B", xlabel="", ylabel="Valor", ylim=(0.70, 0.90))
    axes[0].legend(title="")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.4f", fontsize=7.5, rotation=90, padding=2)

    deltas = comparison.differences[
        comparison.differences["metrica"].isin(
            ["accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"]
        )
    ]
    colors = [TEAL if value > 0 else RED if value < 0 else GREY for value in deltas["diferencia_absoluta"]]
    axes[1].bar(deltas["metrica"], deltas["diferencia_absoluta"], color=colors)
    axes[1].axhline(0, color=NAVY, linewidth=1)
    axes[1].set(title="Diferencia absoluta B − A", xlabel="", ylabel="Δ métrica")
    for x, value in zip(deltas["metrica"], deltas["diferencia_absoluta"]):
        axes[1].annotate(
            f"{value:+.5f}",
            (x, value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    fig.suptitle("Efecto de incorporar la variable de negatividad", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "comparacion_negatividad.png", dpi=180)
    plt.close(fig)
    return catalog, fitted, comparison


# --------------------------------------------------------------------------- #
# Etapa 6 · Selección, persistencia y metadata del modelo definitivo
# --------------------------------------------------------------------------- #
def stage_final_model(
    catalog: pd.DataFrame,
    fitted: dict,
    comparison: NegativityComparison,
    sentiment: pd.DataFrame,
) -> tuple[str, pd.Series]:
    """Aplica la regla de selección, guarda el modelo y documenta su metadata."""

    # La comparación de las cuatro familias preliminares elige la representación
    # base; el modelo definitivo se decide dentro del experimento A/B, que es el
    # único que aísla el efecto de la variable de negatividad.
    preliminary = catalog[
        (catalog["familia"] == "preliminar") & (catalog["modelo"] != "Baseline mayoritaria")
    ].sort_values("f1", ascending=False)
    best_family = str(preliminary.iloc[0]["modelo"])
    candidates = catalog[catalog["familia"] == "reentrenamiento"].reset_index(drop=True)
    winner, reason = select_final_model(candidates)
    row = candidates[candidates["modelo"] == winner].iloc[0]
    model = fitted[winner]
    joblib.dump(model, MODELS / "modelo_final.joblib")

    metadata = {
        "modelo": winner,
        "familia": str(row["familia"]),
        "usa_negatividad": bool(row["usa_negatividad"]),
        "regla_seleccion": "Máximo F1 de la clase desastre; ROC-AUC desempata si la brecha de F1 < 0.002",
        "justificacion": reason,
        "candidatos_evaluados": candidates["modelo"].tolist(),
        "familia_base_elegida": best_family,
        "familia_base_criterio": (
            "Mejor F1 de la clase desastre entre los cuatro modelos preliminares "
            f"({float(preliminary.iloc[0]['f1']):.6f})"
        ),
        "variables": {
            "texto": "clean_text (limpieza de clasificación sobre el tweet crudo)",
            "numerica": "negativity = max(-compound, 0)" if bool(row["usa_negatividad"]) else None,
        },
        "parametros": {
            "tfidf": {key: list(value) if isinstance(value, tuple) else value for key, value in TFIDF_PARAMS.items()},
            "logistic_regression": LOGISTIC_PARAMS,
        },
        "semilla": RANDOM_STATE,
        "particiones": {
            "estrategia": "train_test_split estratificado 80/20",
            "entrenamiento": int(len(comparison.train_indices)),
            "validacion": int(len(comparison.test_indices)),
            "desastre_pct_entrenamiento": float(
                100 * sentiment.iloc[comparison.train_indices]["target"].mean()
            ),
            "desastre_pct_validacion": float(
                100 * sentiment.iloc[comparison.test_indices]["target"].mean()
            ),
        },
        "metricas_validacion": {
            key: (float(row[key]) if key not in {"tn", "fp", "fn", "tp"} else int(row[key]))
            for key in ["accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc", "tn", "fp", "fn", "tp"]
        },
        "dataset": {
            "archivo": "data/raw/train.csv",
            "sha256": "61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df",
            "filas": int(len(sentiment)),
        },
        "artefacto": "models/modelo_final.joblib",
        "generado_por": "scripts/run_final.py",
        "estado": "definitivo",
    }
    (MODELS / "metadata_final.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    matrix = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ConfusionMatrixDisplay(matrix, display_labels=["No desastre", "Desastre"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title(f"Modelo definitivo · {winner}")
    for name, group in comparison.roc_points.groupby("modelo"):
        auc = comparison.metrics.loc[comparison.metrics["modelo"] == name, "roc_auc"].iloc[0]
        axes[1].plot(group["fpr"], group["tpr"], label=f"{name} · AUC {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color=GREY, label="Azar")
    axes[1].set(
        title="Curvas ROC del reentrenamiento",
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos",
    )
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Desempeño del modelo definitivo", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "modelo_final.png", dpi=180)
    plt.close(fig)
    return winner, row


# --------------------------------------------------------------------------- #
# Etapa 7 · Función final sobre texto crudo
# --------------------------------------------------------------------------- #
def stage_prediction_function(model, analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """Documenta el contrato de la función definitiva con ejemplos crudos."""

    examples = [
        "Massive fire near the city, residents are evacuating now! #wildfire https://t.co/abc",
        "That movie was a total disaster lol 😂 @friend",
        "Emergency services report flooding on the highway, call 911 immediately.",
        "I can't believe how great this party is!! 🎉",
        "BREAKING: 7.1 earthquake collapses buildings downtown, rescue teams deployed",
        "My exam results were a train wreck, I'm devastated",
    ]
    table = pd.DataFrame([classify_raw_tweet(model, text, analyzer) for text in examples])
    table.to_csv(TABLES / "ejemplos_funcion_final.csv", index=False)
    return table


# --------------------------------------------------------------------------- #
# Etapa 8 · Análisis de errores del modelo definitivo
# --------------------------------------------------------------------------- #
def stage_error_analysis(
    sentiment: pd.DataFrame, comparison: NegativityComparison, winner: str
) -> pd.DataFrame:
    """Reúne los falsos positivos y falsos negativos con todo su contexto."""

    source = winner if winner in set(comparison.predictions["modelo"]) else MODEL_B
    predictions = comparison.predictions[comparison.predictions["modelo"] == source].copy()
    context = sentiment.reset_index(drop=True)
    merged = predictions.merge(
        context[
            [
                "id",
                "text",
                "clean_text",
                "sentiment_label",
                "sentiment_compound",
                "negativity",
                "keyword",
            ]
        ].reset_index(names="source_index"),
        on="source_index",
        how="left",
    )
    merged["tipo_error"] = np.select(
        [
            (merged["real"] == 0) & (merged["prediccion"] == 1),
            (merged["real"] == 1) & (merged["prediccion"] == 0),
        ],
        ["Falso positivo", "Falso negativo"],
        default="Acierto",
    )
    merged["confianza"] = (merged["probabilidad_desastre"] - 0.5).abs()
    errors = merged[merged["tipo_error"] != "Acierto"].copy()
    columns = [
        "id",
        "tipo_error",
        "text",
        "clean_text",
        "real",
        "prediccion",
        "probabilidad_desastre",
        "sentiment_label",
        "sentiment_compound",
        "negativity",
        "keyword",
    ]
    errors.sort_values(["tipo_error", "confianza"], ascending=[True, False])[columns].to_csv(
        TABLES / "analisis_errores.csv", index=False
    )
    for kind, filename in (
        ("Falso positivo", "errores_falsos_positivos.csv"),
        ("Falso negativo", "errores_falsos_negativos.csv"),
    ):
        subset = errors[errors["tipo_error"] == kind].sort_values("confianza", ascending=False)
        subset.head(15)[columns].to_csv(TABLES / filename, index=False)

    profile = (
        merged.assign(acierto=(merged["real"] == merged["prediccion"]).astype(int))
        .groupby("sentiment_label")
        .agg(
            casos=("id", "size"),
            aciertos=("acierto", "sum"),
            negatividad_media=("negativity", "mean"),
            probabilidad_media=("probabilidad_desastre", "mean"),
        )
        .reset_index()
    )
    profile["tasa_acierto"] = profile["aciertos"] / profile["casos"]
    error_profile = (
        errors.groupby(["tipo_error", "sentiment_label"]).size().rename("casos").reset_index()
    )
    error_profile.to_csv(TABLES / "errores_por_sentimiento.csv", index=False)
    profile.to_csv(TABLES / "acierto_por_sentimiento.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    sns.barplot(
        data=error_profile,
        x="sentiment_label",
        y="casos",
        hue="tipo_error",
        order=["negativo", "neutral", "positivo"],
        palette=[ORANGE, RED],
        ax=axes[0],
    )
    axes[0].set(title="Errores por etiqueta de sentimiento", xlabel="", ylabel="Casos")
    axes[0].legend(title="")
    sns.boxplot(
        data=errors,
        x="tipo_error",
        y="negativity",
        hue="tipo_error",
        palette=[ORANGE, RED],
        legend=False,
        ax=axes[1],
    )
    axes[1].set(title="Negatividad de los errores", xlabel="", ylabel="negativity")
    sns.histplot(
        data=merged,
        x="probabilidad_desastre",
        hue=merged["real"] == merged["prediccion"],
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        palette={True: TEAL, False: RED},
        ax=axes[2],
    )
    axes[2].axvline(0.5, color=NAVY, linestyle="--", linewidth=1)
    axes[2].set(title="Probabilidad estimada: acierto vs error", xlabel="P(desastre)", ylabel="Densidad")
    legend = axes[2].get_legend()
    if legend is not None:
        legend.set_title("Acierto")
    fig.suptitle(f"Análisis de errores · {winner}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "analisis_errores.png", dpi=180)
    plt.close(fig)
    return errors


# --------------------------------------------------------------------------- #
# Etapa 9 · Evidencia de cumplimiento de la rúbrica
# --------------------------------------------------------------------------- #
def stage_rubric_evidence() -> pd.DataFrame:
    """Enlaza cada criterio de la rúbrica con el archivo que lo respalda."""

    rows = [
        ("EDA", 15, [FIGURES / "eda_panorama.png", TABLES / "eda_por_clase.csv"]),
        ("Limpieza y preprocesamiento", 10, [TABLES / "auditoria_limpieza.csv", TABLES / "ejemplos_preprocesamiento.csv"]),
        ("N-gramas, frecuencias y probabilidades", 10, [TABLES / "unigramas_target_1.csv", TABLES / "bigramas_target_1.csv", TABLES / "trigramas_target_1.csv"]),
        ("Modelos clasificadores", 15, [TABLES / "metricas_todos_los_modelos.csv", FIGURES / "modelo_final.png"]),
        ("Función de clasificación", 20, [TABLES / "ejemplos_funcion_final.csv", ROOT / "scripts" / "predict_tweet.py", MODELS / "modelo_final.joblib", MODELS / "metadata_final.json"]),
        ("Sentimiento positivo/negativo/neutral", 10, [TABLES / "sentimiento_distribucion.csv", FIGURES / "sentimiento_final.png"]),
        ("Variable de negatividad", 5, [TABLES / "negatividad_definicion.csv", TABLES / "comparacion_negatividad_diferencias.csv"]),
        ("Resultados y discusión", 15, [TABLES / "analisis_errores.csv", TABLES / "contraste_estadistico.csv", ROOT / "reports" / "informe_final.pdf", ROOT / "notebooks" / "Lab5_Completo.ipynb"]),
    ]
    evidence_rows = []
    for criterion, points, paths in rows:
        relative = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
        missing_paths = [name for name, path in zip(relative, paths) if not path.exists()]
        evidence_rows.append(
            {
                "criterio": criterion,
                "puntos": points,
                "evidencia": "; ".join(relative),
                "generado": "sí" if not missing_paths else "no",
                "faltantes": "; ".join(missing_paths),
            }
        )
    evidence = pd.DataFrame(evidence_rows)
    # El EDA, los n-gramas y las nubes de palabras se generan en run_advance.py.
    # Se verifican aquí porque la evidencia de rúbrica los cita: si faltaran, el
    # informe quedaría apuntando a archivos inexistentes.
    inherited = [
        FIGURES / "eda_panorama.png",
        FIGURES / "ngramas_por_clase.png",
        FIGURES / "nube_palabras_target_0.png",
        FIGURES / "nube_palabras_target_1.png",
        TABLES / "auditoria_limpieza.csv",
        TABLES / "ejemplos_preprocesamiento.csv",
        TABLES / "eda_por_clase.csv",
        TABLES / "unigramas_target_1.csv",
        TABLES / "bigramas_target_1.csv",
        TABLES / "trigramas_target_1.csv",
    ]
    absent = [str(path.relative_to(ROOT)) for path in inherited if not path.exists()]
    if absent:
        raise RuntimeError(
            "Faltan artefactos de EDA y n-gramas que la rúbrica cita: "
            f"{absent}. Ejecute primero: python3 scripts/run_advance.py"
        )

    evidence.to_csv(TABLES / "evidencia_rubrica.csv", index=False, lineterminator="\n")
    return evidence


def main() -> int:
    for directory in (PROCESSED, FIGURES, TABLES, MODELS):
        directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    analyzer = SentimentIntensityAnalyzer()

    raw = pd.read_csv(RAW)
    validate_dataset(raw)
    frame = add_eda_features(raw)
    print(f"[1/9] Dataset validado: {len(frame):,} tweets")

    stopword_audit = stage_stopword_audit(frame)
    content_removed = stopword_audit[stopword_audit["es_palabra_de_contenido"]]
    print(
        f"      Auditoría de palabras vacías: {len(content_removed)} palabras de contenido "
        f"eliminadas por la lista Glasgow"
    )

    sentiment = stage_sentiment(frame)
    counts = sentiment["sentiment_label"].value_counts().to_dict()
    print(f"[2/9] Sentimiento final: {counts}")

    positives, negatives = stage_top_tweets(sentiment)
    print(
        f"[3/9] Top 10 · compound máximo {positives['sentiment_compound'].iloc[0]:.4f}, "
        f"mínimo {negatives['sentiment_compound'].iloc[0]:.4f}"
    )

    contrast = stage_statistics(sentiment)
    row = contrast[contrast["metrica"] == "negativity"].iloc[0]
    print(
        f"[4/9] Mann-Whitney U={row['u_statistic']:.0f} p={row['p_valor_bilateral']:.3e} "
        f"delta de Cliff={row['cliffs_delta']:.4f} ({row['magnitud_efecto']})"
    )

    audit = stage_negativity(sentiment)
    print(f"[5/9] Negatividad en [{sentiment['negativity'].min():.4f}, {sentiment['negativity'].max():.4f}]")

    catalog, fitted, comparison = stage_candidate_models(sentiment)
    delta_f1 = comparison.differences.loc[comparison.differences["metrica"] == "f1", "diferencia_absoluta"].iloc[0]
    delta_auc = comparison.differences.loc[comparison.differences["metrica"] == "roc_auc", "diferencia_absoluta"].iloc[0]
    print(f"[6/9] Reentrenamiento B-A · delta F1={delta_f1:+.6f} delta ROC-AUC={delta_auc:+.6f}")

    winner, winning_row = stage_final_model(catalog, fitted, comparison, sentiment)
    print(
        f"[7/9] Modelo definitivo: {winner} · F1={winning_row['f1']:.6f} "
        f"ROC-AUC={winning_row['roc_auc']:.6f}"
    )

    examples = stage_prediction_function(fitted[winner], analyzer)
    print(f"[8/9] Función final probada sobre {len(examples)} tweets crudos")

    errors = stage_error_analysis(sentiment, comparison, winner)
    evidence = stage_rubric_evidence()
    print(
        f"[9/9] Errores: {int((errors['tipo_error'] == 'Falso positivo').sum())} FP y "
        f"{int((errors['tipo_error'] == 'Falso negativo').sum())} FN · "
        f"rúbrica con {len(evidence)} criterios documentados"
    )

    sentiment.to_csv(PROCESSED / "tweets_sentimiento.csv.gz", index=False, compression="gzip")
    comparison.predictions.to_csv(
        PROCESSED / "predicciones_validacion_final.csv.gz", index=False, compression="gzip"
    )

    summary = {
        "n_tweets": int(len(sentiment)),
        "particion": {
            "entrenamiento": int(len(comparison.train_indices)),
            "validacion": int(len(comparison.test_indices)),
            "semilla": RANDOM_STATE,
        },
        "sentimiento_global": {
            key: int(value) for key, value in sentiment["sentiment_label"].value_counts().items()
        },
        "negatividad": {
            item["indicador"]: item["valor"] for item in audit.to_dict(orient="records")
        },
        "contraste_negatividad": {
            key: (value if isinstance(value, str) else float(value))
            for key, value in contrast[contrast["metrica"] == "negativity"].iloc[0].to_dict().items()
        },
        "modelo_final": {
            "nombre": winner,
            "usa_negatividad": bool(winning_row["usa_negatividad"]),
            **{
                key: (float(winning_row[key]) if key not in {"tn", "fp", "fn", "tp"} else int(winning_row[key]))
                for key in ["accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc", "tn", "fp", "fn", "tp"]
            },
        },
        "efecto_negatividad": {
            item["metrica"]: {
                "a_sin": float(item["modelo_a_sin_negatividad"]),
                "b_con": float(item["modelo_b_con_negatividad"]),
                "diferencia": float(item["diferencia_absoluta"]),
                "sentido": item["sentido"],
            }
            for item in comparison.differences.to_dict(orient="records")
        },
        "errores": {
            "falsos_positivos": int((errors["tipo_error"] == "Falso positivo").sum()),
            "falsos_negativos": int((errors["tipo_error"] == "Falso negativo").sum()),
        },
    }
    (TABLES / "resumen_final.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("\nFlujo final completado. Resultados en outputs/, models/ y data/processed/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
