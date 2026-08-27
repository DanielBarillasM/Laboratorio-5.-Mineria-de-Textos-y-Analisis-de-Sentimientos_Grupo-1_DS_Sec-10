import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from lab5_text.modeling import build_models, predict_tweet


def test_required_model_families_are_available() -> None:
    names = set(build_models())

    assert "Baseline mayoritaria" in names
    assert "Naive Bayes complementario" in names
    assert "Regresión logística" in names
    assert "SVM lineal calibrado" in names


def test_raw_tweet_prediction_contract() -> None:
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("model", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(
        pd.Series(["wildfire evacuation", "earthquake damage", "fun movie", "happy picnic"]),
        [1, 1, 0, 0],
    )
    result = predict_tweet(model, "Evacuation after a wildfire!")

    assert result["clase"] in {"Desastre real", "No desastre"}
    assert 0.0 <= result["probabilidad_desastre"] <= 1.0
    assert result["texto_limpio"]
