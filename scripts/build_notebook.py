"""Construye el notebook narrativo del avance a partir de resultados reproducibles."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Lab5_Avance_75.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11+"},
        "lab5": {"entrega": "avance", "porcentaje": 75, "random_state": 42},
    }
    notebook["cells"] = [
        md(
            r"""
<style>
:root{--ink:#102a43;--blue:#2563eb;--teal:#0f9d91;--soft:#eef5ff;--gold:#f59e0b}
.jp-Notebook{font-family:Inter,Segoe UI,sans-serif;color:var(--ink)}
.hero{padding:38px;border-radius:22px;background:linear-gradient(125deg,#0b1f3a,#164e63 58%,#0f9d91);color:white;box-shadow:0 14px 35px #102a4326}
.hero h1{font-size:2.35rem;margin:.2rem 0}.hero .tag{display:inline-block;padding:7px 13px;border-radius:99px;background:#ffffff20;border:1px solid #ffffff45;font-weight:700}
.team{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:22px}.team div{background:#ffffff14;padding:12px;border-radius:12px}
.section{margin:26px 0 10px;padding:16px 20px;border-left:6px solid var(--blue);border-radius:12px;background:linear-gradient(90deg,var(--soft),#fff)}
.note{padding:14px 18px;border-radius:12px;background:#ecfdf5;border:1px solid #99f6e4}.pending{padding:14px 18px;border-radius:12px;background:#fff7ed;border:1px solid #fed7aa}
table{border-radius:10px;overflow:hidden}.dataframe thead th{background:#16324f!important;color:white!important}.dataframe tbody tr:nth-child(even){background:#f4f8fc}
@media(max-width:800px){.team{grid-template-columns:1fr}}
</style>
<div class="hero">
  <span class="tag">AVANCE REPRODUCIBLE · 75%</span>
  <h1>Laboratorio 5</h1>
  <h2>Minería de Textos y Análisis de Sentimientos</h2>
  <p>Clasificación de tweets sobre desastres reales mediante NLP y aprendizaje automático.</p>
  <div class="team">
    <div><b>Jorge Gabriel Palacios Sales</b><br>231385</div>
    <div><b>Pablo Daniel Barillas Moreno</b><br>22193</div>
    <div><b>Roberto Emiliano Otoniel</b><br>23968</div>
  </div>
  <p><b>Universidad del Valle de Guatemala</b> · Data Science · Sección 10 · Grupo 1</p>
</div>
"""
        ),
        md(
            """
<div class="section"><h2>1 · Alcance y reproducibilidad</h2></div>

Este avance cubre la descripción de los datos, limpieza explicada, exploración, probabilidades de unigramas, bigramas y trigramas, comparación preliminar de modelos, función de inferencia y una primera lectura de sentimiento. Todas las cifras que se muestran provienen de los artefactos generados por `scripts/run_advance.py`; no se escribieron manualmente en el notebook.
"""
        ),
        code(
            """
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Image, display
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path.cwd()
if not (ROOT / "data" / "raw" / "train.csv").exists():
    ROOT = ROOT.parent

TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

# CSS también incrustado desde código para conservar el estilo en ejecución.
display(HTML(
    "<style>"
    ".metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}"
    ".metric{padding:16px;border-radius:14px;background:#f8fbff;border:1px solid #d8e6f5;text-align:center}"
    ".metric b{display:block;font-size:1.55rem;color:#2563eb}.ok{color:#047857;font-weight:700}"
    "</style>"
))
print(f"Raíz del proyecto: {ROOT}")
"""
        ),
        md('<div class="section"><h2>2 · Datos y análisis exploratorio</h2></div>'),
        code(
            """
distribution = pd.read_csv(TABLES / "distribucion_target.csv")
schema = pd.read_csv(TABLES / "dataset_esquema.csv")
display(HTML("<h3>Distribución de la variable objetivo</h3>"))
display(distribution.style.format({"porcentaje": "{:.2f}%"}))
display(HTML("<h3>Esquema y valores faltantes</h3>"))
display(schema)
display(Image(filename=str(FIGURES / "eda_panorama.png"), width=1050))
"""
        ),
        md(
            """
El corpus contiene **7,613 tweets**: 4,342 (57.03%) no describen un desastre real y 3,271 (42.97%) sí. El desbalance es moderado, por lo que se usa partición estratificada y se reportan F1, recall, F1 macro y ROC-AUC además de exactitud. `location` es principalmente un campo auxiliar: su ausencia no impide explotar el texto.
"""
        ),
        code(
            """
display(Image(filename=str(FIGURES / "eda_marcadores_sociales.png"), width=1000))
display(Image(filename=str(FIGURES / "eda_keywords.png"), width=1000))
"""
        ),
        md(
            """
<div class="section"><h2>3 · Preprocesamiento auditable</h2></div>

Se crean dos representaciones con propósitos distintos:

1. **Clasificación:** minúsculas, contracciones expandidas, eliminación de URL y menciones, conservación de la palabra del hashtag, eliminación de puntuación/dígitos y stopwords. Las negaciones (`no`, `not`, `never`) y el token `911` se preservan porque tienen información semántica.
2. **Sentimiento:** limpieza ligera; conserva puntuación, emojis y negaciones para no borrar señales afectivas que VADER necesita.

Esta separación evita el error metodológico de calcular sentimiento sobre texto excesivamente depurado.
"""
        ),
        code(
            """
examples = pd.read_csv(TABLES / "ejemplos_preprocesamiento.csv")
audit = pd.read_csv(TABLES / "auditoria_limpieza.csv")
display(examples[["target", "text", "clean_text"]].head(8))
display(HTML("<h3>Controles automáticos de limpieza</h3>"))
display(audit)
"""
        ),
        md('<div class="section"><h2>4 · N-gramas y probabilidades</h2></div>'),
        code(
            """
display(Image(filename=str(FIGURES / "ngramas_por_clase.png"), width=1100))
display(Image(filename=str(FIGURES / "unigramas_distintivos.png"), width=1000))
"""
        ),
        md(
            r"""
Para cada clase y orden (n), la probabilidad empírica se calcula como

\[
P(g\mid c)=\frac{f(g,c)}{\sum_{g'}f(g',c)}.
\]

No se confunde frecuencia bruta con probabilidad: ambos valores están disponibles en los CSV. Los bigramas como *suicide bomber*, *oil spill* y *northern california* concentran contexto más específico para desastre que palabras aisladas. El cociente logarítmico suavizado complementa el ranking y señala vocabulario distintivo de cada clase.
"""
        ),
        code(
            """
for target, label in [(0, "No desastre"), (1, "Desastre real")]:
    table = pd.read_csv(TABLES / f"bigramas_target_{target}.csv").head(10)
    display(HTML(f"<h3>{label}: diez bigramas principales</h3>"))
    display(table.style.format({"probabilidad": "{:.4%}"}))

display(Image(filename=str(FIGURES / "nube_palabras_target_0.png"), width=850))
display(Image(filename=str(FIGURES / "nube_palabras_target_1.png"), width=850))
"""
        ),
        md('<div class="section"><h2>5 · Modelos preliminares</h2></div>'),
        code(
            """
metrics = pd.read_csv(TABLES / "metricas_modelos_preliminares.csv")
best = metrics.iloc[0]
summary_html = (
    '<div class="metric-grid">'
    f'<div class="metric"><span>Mejor modelo</span><b style="font-size:1rem">{best["modelo"]}</b></div>'
    f'<div class="metric"><span>Exactitud</span><b>{best["accuracy"]:.2%}</b></div>'
    f'<div class="metric"><span>F1 desastre</span><b>{best["f1"]:.3f}</b></div>'
    f'<div class="metric"><span>ROC-AUC</span><b>{best["roc_auc"]:.3f}</b></div>'
    '</div>'
)
display(HTML(summary_html))
display(metrics.style.format({c: "{:.3f}" for c in ["accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"]}))
display(Image(filename=str(FIGURES / "modelos_comparacion.png"), width=1000))
display(Image(filename=str(FIGURES / "modelos_matrices_confusion.png"), width=1100))
display(Image(filename=str(FIGURES / "modelos_curvas_roc.png"), width=900))
"""
        ),
        md(
            """
Los cuatro modelos usan la misma división estratificada 80/20 (`random_state=42`): 6,090 observaciones para entrenamiento y 1,523 para prueba. La **regresión logística** obtiene el mejor F1 (0.781), con recall 0.777 y ROC-AUC 0.867. El SVM calibrado es ligeramente más preciso (0.807) cuando predice desastre, pero deja más falsos negativos. Esta es una selección preliminar; la afinación definitiva forma parte del 25% final.
"""
        ),
        md('<div class="section"><h2>6 · Función para clasificar tweets nuevos</h2></div>'),
        code(
            """
import sys
sys.path.insert(0, str(ROOT / "src"))
from lab5_text.modeling import predict_tweet

model = joblib.load(ROOT / "models" / "modelo_preliminar.joblib")
analyzer = SentimentIntensityAnalyzer()
samples = [
    "Massive fire near the city, residents are evacuating now!",
    "That movie was a total disaster lol 😂",
    "Emergency services report flooding on the highway.",
]
predictions = pd.DataFrame([predict_tweet(model, text, analyzer) for text in samples])
display(predictions.style.format({"probabilidad_desastre": "{:.2%}", "polaridad": "{:.3f}"}))
"""
        ),
        md(
            """
La función recibe **texto crudo**, aplica internamente el preprocesamiento y devuelve una clase comprensible junto con su probabilidad. El ejemplo de la película demuestra por qué la palabra *disaster* por sí sola no basta: el contexto completo disminuye la probabilidad de desastre real.
"""
        ),
        md('<div class="section"><h2>7 · Sentimiento exploratorio</h2></div>'),
        code(
            """
sentiment_summary = pd.read_csv(TABLES / "sentimiento_resumen_por_clase.csv")
display(sentiment_summary.style.format({"mean": "{:.3f}", "median": "{:.3f}", "std": "{:.3f}"}))
display(Image(filename=str(FIGURES / "sentimiento_exploratorio.png"), width=1050))
"""
        ),
        md(
            """
La polaridad media es más negativa en tweets de desastre real (−0.266) que en los que no lo son (−0.052). En este avance se presenta como evidencia descriptiva, no causal: la comparación formal, los diez extremos y el análisis de la variable de negatividad se reservan para la entrega final.
"""
        ),
        md(
            """
<div class="section"><h2>8 · Estado del avance</h2></div>
<div class="note"><b>Completado (75%)</b><br>Datos y EDA; limpieza auditable; unigramas, bigramas y trigramas con probabilidades; nubes de palabras; cuatro modelos comparables; métricas y errores; función de inferencia; sentimiento descriptivo inicial.</div>
<br>
<div class="pending"><b>Reservado para la entrega final (25%)</b><br>Top 10 positivo/negativo con interpretación; contraste estadístico por clase; ingeniería de la variable de negatividad y reentrenamiento; ajuste final, análisis de errores y conclusiones definitivas.</div>

### Conclusión provisional

El texto permite separar tweets de desastres reales con un desempeño sustancialmente superior al baseline. La regresión logística ofrece el mejor equilibrio preliminar y constituye una base transparente para la fase final, en la que se probará si el sentimiento aporta señal predictiva incremental.
"""
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Notebook creado: {OUTPUT}")


if __name__ == "__main__":
    main()
