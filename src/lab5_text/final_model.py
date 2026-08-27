"""Modelo definitivo: reentrenamiento sin y con la variable de negatividad.

El experimento compara dos configuraciones idénticas salvo por una columna:

* **Modelo A** — regresión logística sobre TF-IDF de unigramas y bigramas.
* **Modelo B** — la misma regresión logística sobre TF-IDF más la variable
  numérica ``negativity``.

Controles aplicados para que la comparación sea limpia:

* La misma partición estratificada 80/20 con ``random_state=42``.
* Los mismos hiperparámetros base del vectorizador y del clasificador.
* Todo el ajuste —vocabulario TF-IDF y estandarización de ``negativity``—
  ocurre dentro del ``Pipeline``, por lo que se estima solo con las filas de
  entrenamiento. No hay fuga de información.
* ``negativity`` proviene del lexicón de VADER aplicado al texto del propio
  tweet; es una función determinista del texto y nunca observa la etiqueta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import clean_for_classification, clean_for_sentiment
from .sentiment import negativity_score, sentiment_label


RANDOM_STATE = 42
TEST_SIZE = 0.20
TEXT_COLUMN = "clean_text"
NUMERIC_COLUMN = "negativity"
MODEL_A = "A · TF-IDF sin negatividad"
MODEL_B = "B · TF-IDF con negatividad"
#: Diferencia mínima de F1 de la clase desastre para considerarla decisiva.
F1_TOLERANCE = 0.002

TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.98,
    "sublinear_tf": True,
    "max_features": 35_000,
}
LOGISTIC_PARAMS = {
    "C": 2.0,
    "class_weight": "balanced",
    "max_iter": 2_000,
    "random_state": RANDOM_STATE,
}


@dataclass
class NegativityComparison:
    """Resultado del experimento A/B más los artefactos necesarios aguas abajo."""

    metrics: pd.DataFrame
    differences: pd.DataFrame
    predictions: pd.DataFrame
    fitted_models: dict[str, Pipeline]
    train_indices: np.ndarray
    test_indices: np.ndarray
    roc_points: pd.DataFrame


def build_final_pipeline(*, use_negativity: bool) -> Pipeline:
    """Construye la tubería A o B; solo cambia la presencia del bloque numérico."""

    transformers = [("texto", TfidfVectorizer(**TFIDF_PARAMS), TEXT_COLUMN)]
    if use_negativity:
        transformers.append(("negatividad", StandardScaler(), [NUMERIC_COLUMN]))
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers, remainder="drop")),
            ("model", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )


def stratified_split(target: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Partición 80/20 estratificada y reproducible compartida por ambos modelos."""

    indices = np.arange(len(target))
    return train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )


def score_predictions(y_true: np.ndarray, probability: np.ndarray, *, name: str) -> dict:
    """Calcula el bloque de métricas exigido por la rúbrica."""

    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "modelo": name,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "f1_macro": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compare_negativity(frame: pd.DataFrame) -> NegativityComparison:
    """Entrena A y B sobre la misma partición y cuantifica la diferencia."""

    missing = [column for column in (TEXT_COLUMN, NUMERIC_COLUMN, "target") if column not in frame]
    if missing:
        raise ValueError(f"El marco no tiene las columnas requeridas: {missing}")

    features = frame[[TEXT_COLUMN, NUMERIC_COLUMN]].reset_index(drop=True)
    target = frame["target"].reset_index(drop=True)
    train_idx, test_idx = stratified_split(target)
    x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train = target.iloc[train_idx].to_numpy()
    y_test = target.iloc[test_idx].to_numpy()

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    roc_frames: list[pd.DataFrame] = []
    fitted: dict[str, Pipeline] = {}

    for name, use_negativity in ((MODEL_A, False), (MODEL_B, True)):
        model = build_final_pipeline(use_negativity=use_negativity)
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        metric_rows.append(score_predictions(y_test, probability, name=name))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "source_index": test_idx,
                    "modelo": name,
                    "real": y_test,
                    "prediccion": (probability >= 0.5).astype(int),
                    "probabilidad_desastre": probability,
                }
            )
        )
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
        fitted[name] = model

    metrics = pd.DataFrame(metric_rows)
    metrics["n_train"] = len(train_idx)
    metrics["n_test"] = len(test_idx)
    metrics["usa_negatividad"] = [False, True]

    numeric_columns = [
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
    ]
    row_a = metrics.iloc[0]
    row_b = metrics.iloc[1]
    differences = pd.DataFrame(
        {
            "metrica": numeric_columns,
            "modelo_a_sin_negatividad": [float(row_a[column]) for column in numeric_columns],
            "modelo_b_con_negatividad": [float(row_b[column]) for column in numeric_columns],
        }
    )
    differences["diferencia_absoluta"] = (
        differences["modelo_b_con_negatividad"] - differences["modelo_a_sin_negatividad"]
    )
    differences["sentido"] = np.where(
        differences["metrica"].isin(["fp", "fn"]),
        np.where(
            differences["diferencia_absoluta"] < 0,
            "mejora",
            np.where(differences["diferencia_absoluta"] > 0, "empeora", "sin cambio"),
        ),
        np.where(
            differences["diferencia_absoluta"] > 0,
            "mejora",
            np.where(differences["diferencia_absoluta"] < 0, "empeora", "sin cambio"),
        ),
    )
    return NegativityComparison(
        metrics=metrics,
        differences=differences,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        fitted_models=fitted,
        train_indices=train_idx,
        test_indices=test_idx,
        roc_points=pd.concat(roc_frames, ignore_index=True),
    )


def select_final_model(metrics: pd.DataFrame) -> tuple[str, str]:
    """Aplica la regla de selección documentada y devuelve (modelo, justificación).

    Regla, en este orden:

    1. Prioriza el **F1 de la clase desastre**, que es la clase de interés y la
       que penaliza a la vez falsos positivos y falsos negativos.
    2. Solo los modelos cuyo F1 quede a menos de ``F1_TOLERANCE`` del mejor se
       consideran empatados en la práctica; entre ellos decide el **ROC-AUC**,
       que mide el ordenamiento en todos los umbrales y no solo en 0.5.
    3. Si persiste el empate, gana el nombre menor en orden alfabético, de modo
       que la selección sea determinista y reproducible.

    El ROC-AUC nunca puede rescatar a un modelo con F1 claramente inferior: el
    desempate opera dentro de la banda de tolerancia, no sobre todo el catálogo.
    """

    if len(metrics) < 2:
        raise ValueError("Se requieren al menos dos modelos para seleccionar")
    best_f1 = float(metrics["f1"].max())
    contenders = metrics[metrics["f1"] >= best_f1 - F1_TOLERANCE].copy()
    contenders = contenders.sort_values(
        ["roc_auc", "f1", "modelo"], ascending=[False, False, True]
    ).reset_index(drop=True)
    winner = contenders.iloc[0]
    discarded = metrics[metrics["f1"] < best_f1 - F1_TOLERANCE]

    if len(contenders) == 1:
        gap = best_f1 - float(discarded["f1"].max())
        reason = (
            f"F1 de la clase desastre = {best_f1:.6f}, superior en {gap:.6f} al siguiente "
            f"candidato; la brecha excede la tolerancia de {F1_TOLERANCE}, por lo que el F1 decide "
            f"sin necesidad de desempate."
        )
    else:
        auc_gap = float(winner["roc_auc"] - contenders.iloc[1]["roc_auc"])
        reason = (
            f"{len(contenders)} modelos quedaron dentro de la tolerancia de F1 ({F1_TOLERANCE}); "
            f"entre ellos decide el ROC-AUC = {float(winner['roc_auc']):.6f}, "
            f"mayor por {auc_gap:.6f}."
        )
    return str(winner["modelo"]), reason


def classify_raw_tweet(model: Pipeline, raw_text: str, analyzer=None) -> dict:
    """Clasifica un tweet crudo aplicando exactamente el flujo del entrenamiento.

    Recibe texto sin preprocesar, reproduce la limpieza de clasificación y la de
    sentimiento, calcula polaridad y negatividad, y devuelve el contrato
    completo de la entrega.
    """

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = analyzer or SentimentIntensityAnalyzer()
    text = raw_text if isinstance(raw_text, str) else ""
    clean = clean_for_classification(text)
    compound = float(analyzer.polarity_scores(clean_for_sentiment(text))["compound"])
    negativity = negativity_score(compound)
    row = pd.DataFrame([{TEXT_COLUMN: clean, NUMERIC_COLUMN: negativity}])
    probability = float(model.predict_proba(row)[0, 1])
    return {
        "texto": text,
        "texto_limpio": clean,
        "clase": "Desastre real" if probability >= 0.5 else "No desastre",
        "probabilidad_desastre": probability,
        "sentimiento": sentiment_label(compound),
        "compound": compound,
        "negatividad": negativity,
    }
