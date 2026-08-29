"""Genera la etapa base de EDA, n-gramas y comparación inicial de modelos."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote

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
from wordcloud import WordCloud


sys.path.insert(0, str(ROOT / "src"))

from lab5_text.analysis import (  # noqa: E402
    add_eda_features,
    discriminative_unigrams,
    ngram_frequencies,
    validate_dataset,
)
from lab5_text.modeling import evaluate_models, predict_tweet  # noqa: E402
from lab5_text.sentiment import add_sentiment  # noqa: E402


RAW = ROOT / "data" / "raw" / "train.csv"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
MODELS = ROOT / "models"

BLUE = "#2563eb"
TEAL = "#0f9d91"
ORANGE = "#f59e0b"
NAVY = "#102a43"
RED = "#dc2626"
CLASS_COLORS = {0: BLUE, 1: RED}


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


def save_dataset_products(frame: pd.DataFrame) -> None:
    schema = pd.DataFrame(
        {
            "variable": frame.columns,
            "tipo": [str(frame[column].dtype) for column in frame.columns],
            "faltantes": [int(frame[column].isna().sum()) for column in frame.columns],
            "faltantes_pct": [100 * frame[column].isna().mean() for column in frame.columns],
            "valores_unicos": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
        }
    )
    schema.to_csv(TABLES / "dataset_esquema.csv", index=False)

    target = frame["target"].value_counts().sort_index().rename_axis("target").reset_index(name="tweets")
    target["categoria"] = target["target"].map({0: "No desastre", 1: "Desastre real"})
    target["porcentaje"] = 100 * target["tweets"] / len(frame)
    target.to_csv(TABLES / "distribucion_target.csv", index=False)

    numeric = [
        "char_count",
        "word_count",
        "clean_word_count",
        "unique_word_ratio",
        "url_count",
        "mention_count",
        "hashtag_count",
        "exclamation_count",
        "question_count",
        "digit_count",
        "contains_911",
        "emoji_like_count",
        "keyword_present",
        "location_present",
    ]
    eda = frame.groupby("target")[numeric].agg(["mean", "median", "std"])
    eda.columns = [f"{left}_{right}" for left, right in eda.columns]
    eda.reset_index().to_csv(TABLES / "eda_por_clase.csv", index=False)

    audit = pd.DataFrame(
        [
            {"indicador": "tweets_totales", "valor": len(frame)},
            {"indicador": "texto_original_vacio", "valor": int(frame["text"].eq("").sum())},
            {"indicador": "texto_limpio_vacio", "valor": int(frame["clean_text"].eq("").sum())},
            {"indicador": "palabras_original_promedio", "valor": frame["word_count"].mean()},
            {"indicador": "palabras_limpias_promedio", "valor": frame["clean_word_count"].mean()},
            {"indicador": "tweets_con_url", "valor": int(frame["url_count"].gt(0).sum())},
            {"indicador": "tweets_con_mencion", "valor": int(frame["mention_count"].gt(0).sum())},
            {"indicador": "tweets_con_hashtag", "valor": int(frame["hashtag_count"].gt(0).sum())},
            {"indicador": "tweets_con_911", "valor": int(frame["contains_911"].sum())},
            {
                "indicador": "texto_limpio_con_911",
                "valor": int(frame["clean_text"].str.contains(r"\b911\b", regex=True).sum()),
            },
        ]
    )
    audit.to_csv(TABLES / "auditoria_limpieza.csv", index=False)

    example_mask = (
        frame["url_count"].gt(0)
        | frame["mention_count"].gt(0)
        | frame["hashtag_count"].gt(0)
        | frame["contains_911"].eq(1)
    )
    examples = pd.concat(
        [frame.loc[example_mask].head(8), frame.loc[~example_mask].head(4)], ignore_index=True
    )
    examples[["id", "target", "text", "clean_text"]].to_csv(
        TABLES / "ejemplos_preprocesamiento.csv", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.barplot(data=target, x="categoria", y="tweets", hue="categoria", palette=[BLUE, RED], legend=False, ax=axes[0, 0])
    axes[0, 0].set(title="Distribución de la variable objetivo", xlabel="", ylabel="Tweets")
    for container in axes[0, 0].containers:
        axes[0, 0].bar_label(container, fmt="%.0f")

    missing = frame[["id", "keyword", "location", "text", "target"]].isna().mean().mul(100).sort_values(ascending=False)
    sns.barplot(x=missing.values, y=missing.index, color=ORANGE, ax=axes[0, 1])
    axes[0, 1].set(title="Valores faltantes", xlabel="Porcentaje (%)", ylabel="")

    sns.histplot(data=frame, x="char_count", hue="target", bins=35, element="step", stat="density", common_norm=False, palette=CLASS_COLORS, ax=axes[1, 0])
    axes[1, 0].set(title="Longitud del tweet por clase", xlabel="Caracteres", ylabel="Densidad")
    axes[1, 0].legend(title="Clase", labels=["Desastre", "No desastre"])

    plot_frame = frame.assign(categoria=frame["target"].map({0: "No desastre", 1: "Desastre real"}))
    sns.boxplot(data=plot_frame, x="categoria", y="word_count", hue="categoria", palette=[BLUE, RED], legend=False, showfliers=False, ax=axes[1, 1])
    axes[1, 1].set(title="Cantidad de palabras", xlabel="", ylabel="Palabras por tweet")
    fig.suptitle("Panorama del conjunto de datos", fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_panorama.png", dpi=180)
    plt.close(fig)

    marker_columns = ["url_count", "mention_count", "hashtag_count", "exclamation_count", "question_count", "contains_911", "emoji_like_count"]
    marker_rates = (
        frame.assign(**{column: frame[column].gt(0) for column in marker_columns})
        .groupby("target")[marker_columns]
        .mean()
        .mul(100)
        .T.reset_index(names="marcador")
        .melt(id_vars="marcador", var_name="target", value_name="porcentaje")
    )
    marker_rates["categoria"] = marker_rates["target"].map({0: "No desastre", 1: "Desastre real"})
    marker_rates.to_csv(TABLES / "marcadores_superficie.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.barplot(data=marker_rates, x="marcador", y="porcentaje", hue="categoria", palette=[BLUE, RED], ax=ax)
    ax.set(title="Presencia de marcas sociales antes de limpiar", xlabel="", ylabel="Tweets con marcador (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_marcadores_sociales.png", dpi=180)
    plt.close(fig)


def save_keyword_products(frame: pd.DataFrame) -> None:
    keyword = frame.dropna(subset=["keyword"]).copy()
    keyword["keyword_decoded"] = keyword["keyword"].map(unquote)
    counts = (
        keyword.groupby(["target", "keyword_decoded"]).size().rename("frecuencia").reset_index()
    )
    top = (
        counts.sort_values(["target", "frecuencia"], ascending=[True, False])
        .groupby("target", group_keys=False)
        .head(15)
    )
    top.to_csv(TABLES / "keywords_principales.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, target, title, color in zip(
        axes,
        (0, 1),
        ("No desastre", "Desastre real"),
        (BLUE, RED),
    ):
        subset = top[top["target"] == target].sort_values("frecuencia")
        ax.barh(subset["keyword_decoded"], subset["frecuencia"], color=color)
        ax.set(title=f"Keywords frecuentes · {title}", xlabel="Tweets", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_keywords.png", dpi=180)
    plt.close(fig)


def save_ngram_products(frame: pd.DataFrame) -> None:
    tables: dict[tuple[int, int], pd.DataFrame] = {}
    full_unigrams: dict[int, pd.DataFrame] = {}
    for target in (0, 1):
        texts = frame.loc[frame["target"] == target, "clean_text"]
        for n, label in ((1, "unigramas"), (2, "bigramas"), (3, "trigramas")):
            full = ngram_frequencies(texts, n=n, min_df=2, top_k=None)
            if n == 1:
                full_unigrams[target] = full
            top = full.head(50).copy()
            top["target"] = target
            top.to_csv(TABLES / f"{label}_target_{target}.csv", index=False)
            tables[(n, target)] = top

    distinctive = discriminative_unigrams(full_unigrams[0], full_unigrams[1])
    distinctive = distinctive[distinctive["frecuencia_total"] >= 8]
    distinctive.to_csv(TABLES / "unigramas_distintivos.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    for row, n in enumerate((1, 2, 3)):
        for column, target in enumerate((0, 1)):
            ax = axes[row, column]
            subset = tables[(n, target)].head(15).sort_values("frecuencia")
            ax.barh(subset["ngram"], subset["frecuencia"], color=CLASS_COLORS[target])
            category = "Desastre real" if target == 1 else "No desastre"
            name = {1: "Unigramas", 2: "Bigramas", 3: "Trigramas"}[n]
            ax.set(title=f"{name} · {category}", xlabel="Frecuencia", ylabel="")
    fig.suptitle("N-gramas más frecuentes por clase", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "ngramas_por_clase.png", dpi=180)
    plt.close(fig)

    positive = distinctive.head(15)
    negative = distinctive.tail(15).sort_values("log2_ratio_desastre")
    selected = pd.concat([negative, positive]).sort_values("log2_ratio_desastre")
    colors = [BLUE if value < 0 else RED for value in selected["log2_ratio_desastre"]]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(selected["ngram"], selected["log2_ratio_desastre"], color=colors)
    ax.axvline(0, color=NAVY, linewidth=1)
    ax.set(
        title="Términos distintivos por clase",
        xlabel="log2 razón de frecuencia · positivo favorece desastre",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "unigramas_distintivos.png", dpi=180)
    plt.close(fig)

    for target, color in ((0, BLUE), (1, RED)):
        text = " ".join(frame.loc[frame["target"] == target, "clean_text"])
        cloud = WordCloud(
            width=1400,
            height=800,
            max_words=180,
            background_color="white",
            colormap="Blues" if target == 0 else "Reds",
            random_state=42,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            "Nube de palabras · " + ("No desastre" if target == 0 else "Desastre real"),
            fontsize=16,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(FIGURES / f"nube_palabras_target_{target}.png", dpi=180)
        plt.close(fig)


def save_model_products(frame: pd.DataFrame):
    result = evaluate_models(frame["clean_text"], frame["target"])
    result.metrics.to_csv(TABLES / "metricas_modelos_preliminares.csv", index=False)
    result.predictions.to_csv(PROCESSED / "predicciones_validacion.csv.gz", index=False, compression="gzip")
    result.confusion.to_csv(TABLES / "matrices_confusion.csv", index=False)
    result.roc_points.to_csv(TABLES / "curvas_roc.csv", index=False)
    split = pd.DataFrame(
        [
            {
                "particion": "entrenamiento",
                "filas": len(result.train_indices),
                "desastre_pct": 100 * frame.iloc[result.train_indices]["target"].mean(),
            },
            {
                "particion": "validacion",
                "filas": len(result.test_indices),
                "desastre_pct": 100 * frame.iloc[result.test_indices]["target"].mean(),
            },
        ]
    )
    split.to_csv(TABLES / "particion_modelado.csv", index=False)

    best = result.fitted_models[result.best_model]
    joblib.dump(best, MODELS / "modelo_preliminar.joblib")
    (MODELS / "metadata.json").write_text(
        json.dumps(
            {
                "modelo": result.best_model,
                "filas_entrenamiento": len(result.train_indices),
                "filas_validacion": len(result.test_indices),
                "semilla": 42,
                "representacion": "TF-IDF con unigramas y bigramas",
                "estado": "modelo base para la comparación final",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_metrics = result.metrics[result.metrics["modelo"] != "Baseline mayoritaria"].melt(
        id_vars="modelo",
        value_vars=["precision", "recall", "f1", "roc_auc"],
        var_name="metrica",
        value_name="valor",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_metrics, x="modelo", y="valor", hue="metrica", palette="viridis", ax=ax)
    ax.set(title="Comparación de clasificadores preliminares", xlabel="", ylabel="Métrica", ylim=(0.65, 0.90))
    ax.tick_params(axis="x", rotation=10)
    ax.legend(title="Métrica", ncol=4, loc="lower center")
    fig.tight_layout()
    fig.savefig(FIGURES / "modelos_comparacion.png", dpi=180)
    plt.close(fig)

    model_names = list(result.metrics["modelo"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, name in zip(axes.flat, model_names):
        rows = result.confusion[result.confusion["modelo"] == name]
        matrix = np.zeros((2, 2), dtype=int)
        for row in rows.itertuples():
            matrix[int(row.real), int(row.prediccion)] = int(row.conteo)
        ConfusionMatrixDisplay(matrix, display_labels=["No desastre", "Desastre"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(name)
    fig.suptitle("Matrices de confusión · validación 20%", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "modelos_matrices_confusion.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for name, group in result.roc_points.groupby("modelo"):
        auc = result.metrics.loc[result.metrics["modelo"] == name, "roc_auc"].iloc[0]
        ax.plot(group["fpr"], group["tpr"], label=f"{name} · AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Azar")
    ax.set(title="Curvas ROC en la validación común", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "modelos_curvas_roc.png", dpi=180)
    plt.close(fig)

    analyzer = SentimentIntensityAnalyzer()
    examples = [
        "Massive fire near the city, residents are evacuating now!",
        "That movie was a total disaster lol 😂",
        "Emergency services report flooding on the highway.",
    ]
    predictions = pd.DataFrame([predict_tweet(best, text, analyzer) for text in examples])
    predictions.to_csv(TABLES / "ejemplos_funcion_clasificacion.csv", index=False)
    return result


def save_sentiment_products(frame: pd.DataFrame) -> pd.DataFrame:
    sentiment = add_sentiment(frame)
    summary = (
        sentiment.groupby(["target", "sentiment_label"]).size().rename("tweets").reset_index()
    )
    totals = summary.groupby("target")["tweets"].transform("sum")
    summary["porcentaje"] = 100 * summary["tweets"] / totals
    summary.to_csv(TABLES / "sentimiento_exploratorio.csv", index=False)
    sentiment.groupby("target")["sentiment_compound"].agg(
        ["count", "mean", "median", "std"]
    ).reset_index().to_csv(TABLES / "sentimiento_resumen_por_clase.csv", index=False)

    plot_frame = sentiment.assign(
        categoria=sentiment["target"].map({0: "No desastre", 1: "Desastre real"})
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.barplot(data=summary.assign(categoria=summary["target"].map({0: "No desastre", 1: "Desastre real"})), x="sentiment_label", y="porcentaje", hue="categoria", palette=[BLUE, RED], ax=axes[0])
    axes[0].set(title="Sentimiento preliminar por clase", xlabel="", ylabel="Tweets (%)")
    axes[0].legend(title="")
    sns.violinplot(data=plot_frame, x="categoria", y="sentiment_compound", hue="categoria", palette=[BLUE, RED], legend=False, inner="quart", cut=0, ax=axes[1])
    axes[1].axhline(0, color=NAVY, linewidth=1)
    axes[1].set(title="Polaridad VADER", xlabel="", ylabel="Compound [-1, 1]")
    fig.suptitle("Exploración inicial de sentimiento", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "sentimiento_exploratorio.png", dpi=180)
    plt.close(fig)
    return sentiment


def main() -> int:
    for directory in (PROCESSED, FIGURES, TABLES, MODELS):
        directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    raw = pd.read_csv(RAW)
    validate_dataset(raw)
    frame = add_eda_features(raw)
    save_dataset_products(frame)
    save_keyword_products(frame)
    save_ngram_products(frame)
    results = save_model_products(frame)
    frame = save_sentiment_products(frame)
    frame.to_csv(PROCESSED / "tweets_preprocesados.csv.gz", index=False, compression="gzip")
    print(results.metrics.to_string(index=False))
    print(f"Modelo base seleccionado: {results.best_model}")
    print("Etapa base generada correctamente; continúe con scripts/run_final.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
