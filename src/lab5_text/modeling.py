"""Modelos preliminares y función reproducible de clasificación."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocessing import clean_for_classification, clean_for_sentiment
from .sentiment import sentiment_label


RANDOM_STATE = 42


@dataclass
class ModelResults:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    confusion: pd.DataFrame
    roc_points: pd.DataFrame
    fitted_models: dict[str, Pipeline]
    best_model: str
    train_indices: np.ndarray
    test_indices: np.ndarray


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=35_000,
    )


def build_models() -> dict[str, Pipeline]:
    """Construye baseline y tres clasificadores lineales comparables."""

    return {
        "Baseline mayoritaria": Pipeline(
            [("tfidf", _vectorizer()), ("model", DummyClassifier(strategy="most_frequent"))]
        ),
        "Naive Bayes complementario": Pipeline(
            [("tfidf", _vectorizer()), ("model", ComplementNB(alpha=0.5))]
        ),
        "Regresión logística": Pipeline(
            [
                ("tfidf", _vectorizer()),
                (
                    "model",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVM lineal calibrado": Pipeline(
            [
                ("tfidf", _vectorizer()),
                (
                    "model",
                    CalibratedClassifierCV(
                        LinearSVC(C=0.75, class_weight="balanced", random_state=RANDOM_STATE),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(texts: pd.Series, target: pd.Series) -> ModelResults:
    """Ajusta todos los modelos sobre la misma división estratificada 80/20."""

    indices = np.arange(len(texts))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_train = texts.iloc[train_idx]
    x_test = texts.iloc[test_idx]
    y_train = target.iloc[train_idx].to_numpy()
    y_test = target.iloc[test_idx].to_numpy()

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict] = []
    roc_frames: list[pd.DataFrame] = []
    fitted: dict[str, Pipeline] = {}

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        predicted = (probability >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
        try:
            auc = roc_auc_score(y_test, probability)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, probability)
            roc_frames.append(
                pd.DataFrame(
                    {
                        "modelo": name,
                        "fpr": false_positive_rate,
                        "tpr": true_positive_rate,
                        "umbral": thresholds,
                    }
                )
            )
        except ValueError:
            auc = np.nan
        metric_rows.append(
            {
                "modelo": name,
                "accuracy": accuracy_score(y_test, predicted),
                "precision": precision_score(y_test, predicted, zero_division=0),
                "recall": recall_score(y_test, predicted, zero_division=0),
                "f1": f1_score(y_test, predicted, zero_division=0),
                "f1_macro": f1_score(y_test, predicted, average="macro", zero_division=0),
                "roc_auc": auc,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "source_index": test_idx,
                    "modelo": name,
                    "real": y_test,
                    "prediccion": predicted,
                    "probabilidad_desastre": probability,
                }
            )
        )
        for actual, predicted_label, count in (
            (0, 0, tn),
            (0, 1, fp),
            (1, 0, fn),
            (1, 1, tp),
        ):
            confusion_rows.append(
                {
                    "modelo": name,
                    "real": actual,
                    "prediccion": predicted_label,
                    "conteo": int(count),
                }
            )
        fitted[name] = model

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["f1", "roc_auc", "recall"], ascending=False
    ).reset_index(drop=True)
    best = str(metrics.iloc[0]["modelo"])
    return ModelResults(
        metrics=metrics,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        confusion=pd.DataFrame(confusion_rows),
        roc_points=pd.concat(roc_frames, ignore_index=True),
        fitted_models=fitted,
        best_model=best,
        train_indices=train_idx,
        test_indices=test_idx,
    )


def predict_tweet(model: Pipeline, raw_text: str, sentiment_analyzer=None) -> dict:
    """Clasifica un tweet crudo y devuelve una respuesta lista para mostrar."""

    clean = clean_for_classification(raw_text)
    probability = float(model.predict_proba(pd.Series([clean]))[0, 1])
    result = {
        "texto": raw_text,
        "texto_limpio": clean,
        "clase": "Desastre real" if probability >= 0.5 else "No desastre",
        "probabilidad_desastre": probability,
    }
    if sentiment_analyzer is not None:
        score = sentiment_analyzer.polarity_scores(clean_for_sentiment(raw_text))["compound"]
        result.update({"sentimiento": sentiment_label(score), "polaridad": float(score)})
    return result
