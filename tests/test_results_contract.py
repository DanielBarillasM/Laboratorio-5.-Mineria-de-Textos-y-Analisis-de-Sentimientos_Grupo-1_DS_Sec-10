"""Contrato de los resultados reproducibles que genera scripts/run_final.py.

Estas pruebas no reentrenan nada: verifican que los archivos publicados sean
coherentes entre sí y con lo que afirman el notebook y el informe. Si el flujo
final aún no se ha ejecutado, se omiten con un mensaje explícito.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "models"

REQUIRED_TABLES = [
    "sentimiento_distribucion.csv",
    "sentimiento_estadisticos_por_clase.csv",
    "sentimiento_umbrales.csv",
    "top10_positivos.csv",
    "top10_negativos.csv",
    "top10_patrones.csv",
    "contraste_estadistico.csv",
    "negatividad_definicion.csv",
    "comparacion_negatividad_metricas.csv",
    "comparacion_negatividad_diferencias.csv",
    "metricas_todos_los_modelos.csv",
    "analisis_errores.csv",
    "ejemplos_funcion_final.csv",
    "evidencia_rubrica.csv",
]
REQUIRED_FIGURES = [
    "sentimiento_final.png",
    "top10_composicion.png",
    "contraste_negatividad.png",
    "comparacion_negatividad.png",
    "modelo_final.png",
    "analisis_errores.png",
]

pytestmark = pytest.mark.skipif(
    not (TABLES / "resumen_final.json").exists(),
    reason="Ejecute primero: python3 scripts/run_final.py",
)


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((TABLES / "resumen_final.json").read_text(encoding="utf-8"))


def test_every_required_table_and_figure_was_generated() -> None:
    missing = [name for name in REQUIRED_TABLES if not (TABLES / name).exists()]
    missing += [name for name in REQUIRED_FIGURES if not (FIGURES / name).exists()]

    assert missing == []


def test_the_final_model_and_its_metadata_are_persisted() -> None:
    assert (MODELS / "modelo_final.joblib").exists()
    metadata = json.loads((MODELS / "metadata_final.json").read_text(encoding="utf-8"))
    for key in ("modelo", "usa_negatividad", "regla_seleccion", "justificacion", "semilla",
                "particiones", "metricas_validacion", "parametros", "variables"):
        assert key in metadata, key
    assert metadata["semilla"] == 42
    assert metadata["estado"] == "definitivo"


def test_the_split_is_the_documented_stratified_eighty_twenty() -> None:
    metadata = json.loads((MODELS / "metadata_final.json").read_text(encoding="utf-8"))
    partitions = metadata["particiones"]

    assert partitions["entrenamiento"] == 6090
    assert partitions["validacion"] == 1523
    assert partitions["entrenamiento"] + partitions["validacion"] == 7613
    assert partitions["desastre_pct_entrenamiento"] == pytest.approx(
        partitions["desastre_pct_validacion"], abs=0.5
    )


def test_sentiment_distribution_covers_the_three_labels_per_class() -> None:
    distribution = pd.read_csv(TABLES / "sentimiento_distribucion.csv")

    assert set(distribution["sentiment_label"]) == {"negativo", "neutral", "positivo"}
    assert set(distribution["target"]) == {0, 1}
    assert distribution["tweets"].sum() == 7613
    for _, group in distribution.groupby("target"):
        assert group["porcentaje"].sum() == pytest.approx(100.0)


def test_negativity_stays_inside_its_range_across_the_whole_corpus() -> None:
    audit = pd.read_csv(TABLES / "negatividad_definicion.csv").set_index("indicador")["valor"]

    assert audit["definicion"] == "negativity = max(-compound, 0)"
    assert 0.0 <= float(audit["minimo_observado"]) <= 1.0
    assert 0.0 <= float(audit["maximo_observado"]) <= 1.0
    assert float(audit["minimo_observado"]) <= float(audit["maximo_observado"])


def test_the_top_ten_tables_have_ten_rows_and_the_required_columns() -> None:
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
    positives = pd.read_csv(TABLES / "top10_positivos.csv")
    negatives = pd.read_csv(TABLES / "top10_negativos.csv")

    assert len(positives) == len(negatives) == 10
    assert required.issubset(positives.columns) and required.issubset(negatives.columns)
    assert (positives["sentiment_label"] == "positivo").all()
    assert (negatives["sentiment_label"] == "negativo").all()
    assert (negatives["negativity"] > positives["negativity"].max()).all()


def test_the_statistical_contrast_answers_the_question_with_an_effect_size() -> None:
    contrast = pd.read_csv(TABLES / "contraste_estadistico.csv")
    row = contrast[contrast["metrica"] == "negativity"].iloc[0]

    assert row["n_grupo_1"] + row["n_grupo_0"] == 7613
    assert 0.0 <= row["p_valor_bilateral"] <= 1.0
    assert -1.0 <= row["cliffs_delta"] <= 1.0
    assert row["magnitud_efecto"] in {"insignificante", "pequeño", "mediano", "grande"}
    assert row["ic95_dif_medias_inferior"] <= row["diferencia_medias"] <= row["ic95_dif_medias_superior"]


def test_the_negativity_experiment_reports_a_signed_difference_per_metric() -> None:
    differences = pd.read_csv(TABLES / "comparacion_negatividad_diferencias.csv")

    assert set(differences["metrica"]) == {
        "accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc", "tn", "fp", "fn", "tp"
    }
    computed = differences["modelo_b_con_negatividad"] - differences["modelo_a_sin_negatividad"]
    assert (differences["diferencia_absoluta"] - computed).abs().max() == pytest.approx(0.0, abs=1e-9)
    assert set(differences["sentido"]) <= {"mejora", "empeora", "sin cambio"}


def test_the_reported_conclusion_matches_the_measured_difference(summary: dict) -> None:
    """El modelo final debe ser el que la evidencia respalda, no el preferido."""

    effect = summary["efecto_negatividad"]
    final = summary["modelo_final"]
    delta_f1 = effect["f1"]["diferencia"]

    if delta_f1 < -0.002:
        assert final["usa_negatividad"] is False
        assert final["f1"] == pytest.approx(effect["f1"]["a_sin"])
    elif delta_f1 > 0.002:
        assert final["usa_negatividad"] is True
        assert final["f1"] == pytest.approx(effect["f1"]["b_con"])
    else:
        assert final["f1"] == pytest.approx(
            max(effect["f1"]["a_sin"], effect["f1"]["b_con"]), abs=0.002
        )

    expected_direction = "mejora" if delta_f1 > 0 else "empeora" if delta_f1 < 0 else "sin cambio"
    assert effect["f1"]["sentido"] == expected_direction


def test_the_confusion_matrix_of_every_model_covers_the_validation_set() -> None:
    catalog = pd.read_csv(TABLES / "metricas_todos_los_modelos.csv")

    assert len(catalog) >= 6
    totals = catalog["tn"] + catalog["fp"] + catalog["fn"] + catalog["tp"]
    assert (totals == 1523).all()
    assert catalog["roc_auc"].between(0.0, 1.0).all()
    assert catalog["f1"].between(0.0, 1.0).all()


def test_error_analysis_carries_the_context_needed_to_interpret_it() -> None:
    errors = pd.read_csv(TABLES / "analisis_errores.csv")
    required = {
        "id", "tipo_error", "text", "clean_text", "real", "prediccion",
        "probabilidad_desastre", "sentiment_label", "sentiment_compound", "negativity",
    }

    assert required.issubset(errors.columns)
    assert set(errors["tipo_error"]) == {"Falso positivo", "Falso negativo"}
    assert (errors["real"] != errors["prediccion"]).all()
    false_positives = errors[errors["tipo_error"] == "Falso positivo"]
    false_negatives = errors[errors["tipo_error"] == "Falso negativo"]
    assert (false_positives["probabilidad_desastre"] >= 0.5).all()
    assert (false_negatives["probabilidad_desastre"] < 0.5).all()


def test_error_counts_agree_with_the_final_confusion_matrix(summary: dict) -> None:
    errors = pd.read_csv(TABLES / "analisis_errores.csv")
    final = summary["modelo_final"]

    assert int((errors["tipo_error"] == "Falso positivo").sum()) == final["fp"]
    assert int((errors["tipo_error"] == "Falso negativo").sum()) == final["fn"]
    assert final["tn"] + final["fp"] + final["fn"] + final["tp"] == 1523


def test_the_prediction_function_examples_honour_the_public_contract() -> None:
    examples = pd.read_csv(TABLES / "ejemplos_funcion_final.csv")
    required = {
        "texto", "texto_limpio", "clase", "probabilidad_desastre",
        "sentimiento", "compound", "negatividad",
    }

    assert required.issubset(examples.columns)
    assert len(examples) >= 5
    assert examples["probabilidad_desastre"].between(0.0, 1.0).all()
    assert examples["negatividad"].between(0.0, 1.0).all()
    assert set(examples["clase"]) <= {"Desastre real", "No desastre"}
    assert set(examples["sentimiento"]) <= {"negativo", "neutral", "positivo"}


def test_the_rubric_evidence_accounts_for_the_full_hundred_points() -> None:
    evidence = pd.read_csv(TABLES / "evidencia_rubrica.csv")

    assert evidence["puntos"].sum() == 100
    assert len(evidence) == 8
    assert evidence["evidencia"].str.len().gt(0).all()
