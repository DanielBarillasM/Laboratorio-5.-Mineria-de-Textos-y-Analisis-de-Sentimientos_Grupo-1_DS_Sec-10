"""Contrato del análisis de sentimiento: etiquetas, negatividad y conteos."""

import pandas as pd
import pytest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from lab5_text.sentiment import (
    NEGATIVE_THRESHOLD,
    POSITIVE_THRESHOLD,
    add_sentiment,
    count_sentiment_words,
    extreme_tweets,
    negativity_score,
    sentiment_label,
    sentiment_summary,
)


@pytest.fixture(scope="module")
def analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "keyword": [None] * 6,
            "location": [None] * 6,
            "text": [
                "Terrible earthquake destroyed the whole town, people are dying",
                "What a wonderful and happy day, I love everything :)",
                "The train arrives at nine",
                "Massive flood, families lost their homes, devastating",
                "Great concert last night, amazing show!",
                "Report filed at the office",
            ],
            "target": [1, 0, 0, 1, 0, 0],
        }
    )


@pytest.mark.parametrize(
    ("compound", "expected"),
    [
        (1.0, "positivo"),
        (POSITIVE_THRESHOLD, "positivo"),
        (0.049, "neutral"),
        (0.0, "neutral"),
        (-0.049, "neutral"),
        (NEGATIVE_THRESHOLD, "negativo"),
        (-1.0, "negativo"),
    ],
)
def test_sentiment_labels_follow_documented_thresholds(compound: float, expected: str) -> None:
    assert sentiment_label(compound) == expected


@pytest.mark.parametrize("compound", [-1.0, -0.5, -0.05, 0.0, 0.05, 0.5, 1.0])
def test_negativity_stays_inside_the_unit_interval(compound: float) -> None:
    value = negativity_score(compound)

    assert 0.0 <= value <= 1.0


def test_negativity_is_zero_unless_the_tweet_is_negative() -> None:
    assert negativity_score(0.0) == 0.0
    assert negativity_score(0.87) == 0.0
    assert negativity_score(-0.6) == pytest.approx(0.6)
    # Valores más negativos deben producir mayor negatividad.
    assert negativity_score(-0.9) > negativity_score(-0.3)


def test_word_counts_partition_every_token(analyzer: SentimentIntensityAnalyzer) -> None:
    counts = count_sentiment_words(analyzer, "Terrible fire but the rescue was great")

    assert counts["pos_word_count"] + counts["neg_word_count"] + counts["neu_word_count"] == counts["token_count"]
    assert counts["token_count"] > 0


def test_word_counts_detect_polarity_in_both_directions(analyzer: SentimentIntensityAnalyzer) -> None:
    negative = count_sentiment_words(analyzer, "awful terrible horrible disaster")
    positive = count_sentiment_words(analyzer, "wonderful amazing great happy")
    empty = count_sentiment_words(analyzer, "")

    assert negative["neg_word_count"] >= 4 and negative["pos_word_count"] == 0
    assert positive["pos_word_count"] >= 4 and positive["neg_word_count"] == 0
    assert empty == {"pos_word_count": 0, "neg_word_count": 0, "neu_word_count": 0, "token_count": 0}


def test_add_sentiment_returns_the_documented_contract(corpus: pd.DataFrame) -> None:
    enriched = add_sentiment(corpus)
    expected = {
        "sentiment_text",
        "sentiment_compound",
        "sentiment_negative",
        "sentiment_positive",
        "sentiment_neutral",
        "sentiment_label",
        "negativity",
        "pos_word_count",
        "neg_word_count",
        "neu_word_count",
        "token_count",
    }

    assert expected.issubset(enriched.columns)
    assert len(enriched) == len(corpus)
    assert enriched["negativity"].between(0.0, 1.0).all()
    assert enriched["sentiment_compound"].between(-1.0, 1.0).all()
    assert set(enriched["sentiment_label"]).issubset({"negativo", "neutral", "positivo"})


def test_negativity_matches_the_label_it_comes_from(corpus: pd.DataFrame) -> None:
    enriched = add_sentiment(corpus)
    negatives = enriched[enriched["sentiment_label"] == "negativo"]
    others = enriched[enriched["sentiment_label"] != "negativo"]

    assert (negatives["negativity"] > 0).all()
    assert (others["negativity"] == 0).all()


def test_sentiment_summary_percentages_add_up_by_class(corpus: pd.DataFrame) -> None:
    summary = sentiment_summary(add_sentiment(corpus))

    assert set(summary["sentiment_label"]) == {"negativo", "neutral", "positivo"}
    for _, group in summary.groupby("target"):
        assert group["porcentaje"].sum() == pytest.approx(100.0)


def test_extreme_tweets_are_ordered_and_carry_the_required_columns(corpus: pd.DataFrame) -> None:
    enriched = add_sentiment(corpus)
    positives, negatives = extreme_tweets(enriched, k=3)
    required = {
        "id",
        "text",
        "target",
        "categoria",
        "sentiment_label",
        "sentiment_compound",
        "negativity",
        "pos_word_count",
        "neg_word_count",
    }

    assert list(positives.columns) == list(negatives.columns)
    assert required.issubset(positives.columns)
    assert len(positives) == len(negatives) == 3
    assert positives["sentiment_compound"].is_monotonic_decreasing
    assert negatives["sentiment_compound"].is_monotonic_increasing
    assert positives["sentiment_compound"].iloc[0] >= negatives["sentiment_compound"].iloc[0]
