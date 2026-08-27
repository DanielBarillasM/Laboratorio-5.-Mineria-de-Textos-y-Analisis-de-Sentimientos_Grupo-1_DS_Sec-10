"""Análisis de sentimiento con VADER, calibrado para el registro de Twitter.

Decisiones documentadas
-----------------------
* Umbrales oficiales de Hutto y Gilbert (2014): ``compound >= 0.05`` es
  positivo, ``compound <= -0.05`` es negativo y el intervalo abierto
  intermedio es neutral.
* El texto que entra a VADER se limpia con :func:`clean_for_sentiment`, que
  conserva emojis, puntuación, mayúsculas y negaciones porque son las señales
  que el algoritmo pondera; solo se retiran URL y menciones, que no aportan
  polaridad.
* La variable principal de negatividad es ``negativity = max(-compound, 0)``.
  Queda acotada en [0, 1], vale 0 para todo tweet neutro o positivo y crece
  con la intensidad negativa. Se conserva además la componente ``neg`` de
  VADER como medida secundaria de contraste.
* El conteo de palabras positivas, negativas y neutrales es de nivel léxico:
  se consulta la valencia de cada token en el lexicón de VADER sin aplicar
  negaciones ni intensificadores. Esos modificadores sí intervienen en
  ``compound``, por lo que ambas medidas son complementarias, no redundantes.
"""

from __future__ import annotations

import string

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer, SentiText

from .preprocessing import clean_for_sentiment


POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
LABELS = ("negativo", "neutral", "positivo")
PUNCTUATION = string.punctuation


def sentiment_label(compound: float) -> str:
    """Traduce el puntaje compuesto a una etiqueta legible."""

    if compound >= POSITIVE_THRESHOLD:
        return "positivo"
    if compound <= NEGATIVE_THRESHOLD:
        return "negativo"
    return "neutral"


def negativity_score(compound: float) -> float:
    """Define la variable principal de negatividad en el rango [0, 1]."""

    value = -float(compound)
    return float(value) if value > 0.0 else 0.0


def _token_valence(analyzer: SentimentIntensityAnalyzer, token: str) -> float:
    """Valencia léxica de un token, resolviendo emojis y puntuación adherida."""

    if token in analyzer.emojis:
        description = analyzer.emojis[token]
        parts = [analyzer.lexicon.get(word, 0.0) for word in description.split()]
        return max(parts, key=abs, default=0.0)
    lowered = token.lower()
    if lowered in analyzer.lexicon:
        return float(analyzer.lexicon[lowered])
    stripped = lowered.strip(PUNCTUATION)
    if stripped and stripped in analyzer.lexicon:
        return float(analyzer.lexicon[stripped])
    return 0.0


def count_sentiment_words(analyzer: SentimentIntensityAnalyzer, text: str) -> dict[str, int]:
    """Cuenta tokens positivos, negativos y neutrales según el lexicón de VADER."""

    tokens = SentiText(text).words_and_emoticons if text.strip() else []
    positive = negative = neutral = 0
    for token in tokens:
        valence = _token_valence(analyzer, token)
        if valence > 0:
            positive += 1
        elif valence < 0:
            negative += 1
        else:
            neutral += 1
    return {
        "pos_word_count": positive,
        "neg_word_count": negative,
        "neu_word_count": neutral,
        "token_count": len(tokens),
    }


def add_sentiment(
    frame: pd.DataFrame,
    *,
    analyzer: SentimentIntensityAnalyzer | None = None,
) -> pd.DataFrame:
    """Añade polaridad, etiqueta, negatividad y conteos léxicos por tweet."""

    analyzer = analyzer or SentimentIntensityAnalyzer()
    result = frame.copy()
    result["sentiment_text"] = result["text"].map(clean_for_sentiment)
    scores = result["sentiment_text"].map(analyzer.polarity_scores)
    result["sentiment_compound"] = scores.map(lambda value: value["compound"]).astype(float)
    result["sentiment_negative"] = scores.map(lambda value: value["neg"]).astype(float)
    result["sentiment_positive"] = scores.map(lambda value: value["pos"]).astype(float)
    result["sentiment_neutral"] = scores.map(lambda value: value["neu"]).astype(float)
    result["sentiment_label"] = result["sentiment_compound"].map(sentiment_label)
    result["negativity"] = result["sentiment_compound"].map(negativity_score)
    counts = pd.DataFrame(
        result["sentiment_text"].map(lambda text: count_sentiment_words(analyzer, text)).tolist(),
        index=result.index,
    )
    return pd.concat([result, counts], axis=1)


def sentiment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Reparte los tweets por clase y etiqueta de sentimiento, con porcentajes."""

    summary = (
        frame.groupby(["target", "sentiment_label"])
        .size()
        .rename("tweets")
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [sorted(frame["target"].unique()), LABELS], names=["target", "sentiment_label"]
    ).to_frame(index=False)
    summary = complete.merge(summary, on=["target", "sentiment_label"], how="left").fillna({"tweets": 0})
    summary["tweets"] = summary["tweets"].astype(int)
    summary["porcentaje"] = 100 * summary["tweets"] / summary.groupby("target")["tweets"].transform("sum")
    summary["categoria"] = summary["target"].map({0: "No desastre", 1: "Desastre real"})
    return summary


def extreme_tweets(frame: pd.DataFrame, *, k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve los k tweets más positivos y los k más negativos, en ese orden.

    El desempate usa ``id`` para que la tabla sea reproducible ante empates de
    ``compound``, frecuentes en los extremos de la escala.
    """

    columns = [
        "id",
        "text",
        "target",
        "categoria",
        "sentiment_label",
        "sentiment_compound",
        "negativity",
        "pos_word_count",
        "neg_word_count",
    ]
    table = frame.assign(categoria=frame["target"].map({0: "No desastre", 1: "Desastre real"}))
    positives = table.sort_values(["sentiment_compound", "id"], ascending=[False, True]).head(k)
    negatives = table.sort_values(["sentiment_compound", "id"], ascending=[True, True]).head(k)
    return (
        positives[columns].reset_index(drop=True),
        negatives[columns].reset_index(drop=True),
    )
