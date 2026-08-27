"""Herramientas reproducibles para el Laboratorio 5 de minería de textos."""

from .analysis import add_eda_features, discriminative_unigrams, ngram_frequencies, validate_dataset
from .final_model import (
    build_final_pipeline,
    classify_raw_tweet,
    compare_negativity,
    select_final_model,
    stratified_split,
)
from .modeling import build_models, evaluate_models, predict_tweet
from .preprocessing import clean_for_classification, clean_for_sentiment, surface_markers
from .sentiment import (
    add_sentiment,
    count_sentiment_words,
    extreme_tweets,
    negativity_score,
    sentiment_label,
    sentiment_summary,
)
from .statistics import bootstrap_difference_ci, cliffs_delta, cliffs_delta_magnitude, compare_groups

__all__ = [
    "add_eda_features",
    "add_sentiment",
    "bootstrap_difference_ci",
    "build_final_pipeline",
    "build_models",
    "classify_raw_tweet",
    "clean_for_classification",
    "clean_for_sentiment",
    "cliffs_delta",
    "cliffs_delta_magnitude",
    "compare_groups",
    "compare_negativity",
    "count_sentiment_words",
    "discriminative_unigrams",
    "evaluate_models",
    "extreme_tweets",
    "negativity_score",
    "ngram_frequencies",
    "predict_tweet",
    "select_final_model",
    "sentiment_label",
    "sentiment_summary",
    "stratified_split",
    "surface_markers",
    "validate_dataset",
]
