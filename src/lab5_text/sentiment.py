"""Exploración inicial de sentimiento con VADER, diseñada para tweets."""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocessing import clean_for_sentiment


def sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "positivo"
    if compound <= -0.05:
        return "negativo"
    return "neutral"


def add_sentiment(frame: pd.DataFrame) -> pd.DataFrame:
    """Añade polaridad VADER sin eliminar emoticones, puntuación ni negaciones."""

    analyzer = SentimentIntensityAnalyzer()
    result = frame.copy()
    result["sentiment_text"] = result["text"].map(clean_for_sentiment)
    scores = result["sentiment_text"].map(analyzer.polarity_scores)
    result["sentiment_compound"] = scores.map(lambda value: value["compound"])
    result["sentiment_negative"] = scores.map(lambda value: value["neg"])
    result["sentiment_positive"] = scores.map(lambda value: value["pos"])
    result["sentiment_neutral"] = scores.map(lambda value: value["neu"])
    result["sentiment_label"] = result["sentiment_compound"].map(sentiment_label)
    return result
