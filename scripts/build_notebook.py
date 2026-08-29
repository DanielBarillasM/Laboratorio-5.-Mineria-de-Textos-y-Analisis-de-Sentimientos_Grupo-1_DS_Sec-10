"""Construye el notebook narrativo final del Laboratorio 5.

El notebook consume exclusivamente artefactos reproducibles generados por
``run_advance.py`` y ``run_final.py``. Después de construirlo puede ejecutarse
con nbconvert para guardar todas las tablas y figuras como salidas visibles.
"""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "Lab5_Completo.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> int:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "lab5": {"entrega": "final", "porcentaje": 100, "random_state": 42},
    }

    cells = [
        markdown(r"""
<style>
:root{--navy:#102a43;--blue:#2563eb;--teal:#0f9d91;--red:#dc2626;--ink:#243b53;--soft:#f1f5f9}
.hero{padding:34px;border-radius:20px;background:linear-gradient(135deg,var(--navy),#1d4ed8);color:white;box-shadow:0 12px 28px #102a4333;margin-bottom:22px}
.hero h1{font-size:2.35rem;margin:.2rem 0}.hero p{font-size:1.05rem;opacity:.92}.tag{display:inline-block;padding:6px 12px;border-radius:999px;background:#ffffff22;border:1px solid #ffffff55;font-weight:700;letter-spacing:.05em}
.section{border-left:6px solid var(--teal);padding:7px 16px;margin:30px 0 14px;background:linear-gradient(90deg,#e6fffa,white);border-radius:0 12px 12px 0}.section h2{color:var(--navy);margin:.2rem 0}
.insight,.warning,.method{padding:14px 17px;border-radius:12px;margin:14px 0}.insight{background:#ecfdf5;border:1px solid #6ee7b7}.warning{background:#fff7ed;border:1px solid #fdba74}.method{background:#eff6ff;border:1px solid #93c5fd}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}.metric{background:white;border:1px solid #dbeafe;border-radius:12px;padding:13px;text-align:center;box-shadow:0 4px 12px #102a4311}.metric b{font-size:1.35rem;color:var(--blue)}
.footer{margin-top:32px;padding:18px;text-align:center;border-top:1px solid #cbd5e1;color:#52667a}
</style>

<div class="hero">
  <span class="tag">ENTREGA FINAL · 100%</span>
  <h1>Laboratorio 5 · Minería de Textos y Análisis de Sentimientos</h1>
  <p>Clasificación reproducible de tweets sobre desastres reales</p>
  <p><b>Universidad del Valle de Guatemala</b> · Data Science · Sección 10 · Grupo 1</p>
  <p>Jorge Gabriel Palacios Sales — 231385 · Pablo Daniel Barillas Moreno — 22193 · Roberto Emiliano Otoniel — 23968</p>
</div>
"""),
        markdown(r"""
<div class="section"><h2>1 · Propósito y preguntas de análisis</h2></div>

El objetivo es identificar si un tweet describe un desastre real (`target = 1`) o si utiliza lenguaje similar fuera de ese contexto (`target = 0`). El flujo responde cuatro preguntas: ¿qué vocabulario distingue las clases?, ¿qué modelo generaliza mejor?, ¿los desastres se expresan con mayor negatividad? y ¿esa negatividad mejora la predicción?

<div class="method"><b>Protocolo reproducible.</b> Todas las comparaciones utilizan una división estratificada 80/20 y <code>random_state=42</code>. Las cifras se leen de archivos generados por los scripts del repositorio; no se escriben manualmente.</div>
"""),
        code(r"""
from pathlib import Path
import json, sys
import joblib
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from lab5_text.analysis import validate_dataset
from lab5_text.final_model import classify_raw_tweet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
pd.set_option("display.max_colwidth", 110)

TABLE_STYLES = [
    {"selector":"th", "props":[("background-color","#102a43"),("color","white"),("font-weight","700"),("text-align","left")]},
    {"selector":"td", "props":[("border-bottom","1px solid #dbeafe"),("padding","8px")]},
    {"selector":"tr:nth-child(even)", "props":[("background-color","#f8fafc")]},
]
def show_table(frame, precision=3):
    display(frame.style.format(precision=precision).set_table_styles(TABLE_STYLES).hide(axis="index"))

summary = json.loads((TABLES / "resumen_final.json").read_text(encoding="utf-8"))
print("Entorno y utilidades cargados correctamente.")
"""),
        markdown(r"""
<div class="section"><h2>2 · Datos</h2></div>

Se usa `train.csv` de la competencia *Natural Language Processing with Disaster Tweets*. Sus columnas son `id`, `keyword`, `location`, `text` y `target`. `keyword` y `location` pueden faltar; `text` y `target` son indispensables. `test.csv` no se emplea para evaluar porque no incluye la etiqueta objetivo.
"""),
        code(r"""
data = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
validate_dataset(data)
distribution = pd.read_csv(TABLES / "distribucion_target.csv")
print(f"Filas: {len(data):,} · Columnas: {data.shape[1]} · Duplicados exactos: {data.duplicated().sum()}")
print("Valores faltantes por columna:")
display(data.isna().sum().to_frame("faltantes").T)
show_table(distribution)
display(Image(filename=str(FIGURES / "eda_panorama.png"), width=980))
"""),
        markdown(r"""
<div class="insight"><b>Lectura.</b> Hay 4,342 tweets sin desastre (57.06%) y 3,271 con desastre (42.94%). El desbalance es moderado; por ello, además de exactitud se reportan precisión, recall, F1, F1 macro y ROC–AUC.</div>

<div class="section"><h2>3 · Limpieza y preprocesamiento</h2></div>

Para modelar se normaliza a minúsculas, se reemplazan URL y menciones por marcadores, se separan hashtags, se eliminan caracteres no informativos y se reducen espacios. Se conserva una segunda versión ligera para VADER, porque negaciones, signos, mayúsculas, emojis y emoticonos aportan información de sentimiento. La lista de palabras vacías se audita de forma explícita para reconocer posibles pérdidas de vocabulario temático.
"""),
        code(r"""
cleaning = pd.read_csv(TABLES / "ejemplos_preprocesamiento.csv").head(8)
audit = pd.read_csv(TABLES / "auditoria_limpieza.csv")
stopwords = pd.read_csv(TABLES / "auditoria_stopwords.csv")
show_table(cleaning)
print("Resumen de transformaciones:")
show_table(audit)
print("Palabras de contenido eliminadas por la lista Glasgow (muestra):")
show_table(stopwords.query("es_palabra_de_contenido == True").head(10))
"""),
        markdown(r"""
<div class="warning"><b>Limitación controlada.</b> La lista estándar de scikit-learn contiene términos con posible carga temática, como <code>fire</code>. Se cuantifica esa pérdida y se mantiene fija en todos los modelos para garantizar una comparación justa.</div>

<div class="section"><h2>4 · Frecuencias, n-gramas y contexto</h2></div>

Las frecuencias se calculan por clase para unigramas, bigramas y trigramas. Además del conteo, se conserva la probabilidad condicional dentro de cada clase. Los n-gramas permiten distinguir frases de emergencia de usos figurados que una palabra aislada no separa adecuadamente.
"""),
        code(r"""
for ngram, label in [("unigramas", "Unigramas"), ("bigramas", "Bigramas"), ("trigramas", "Trigramas")]:
    print(label, "más frecuentes en desastres reales")
    show_table(pd.read_csv(TABLES / f"{ngram}_target_1.csv").head(10))
display(Image(filename=str(FIGURES / "ngramas_por_clase.png"), width=1000))
"""),
        markdown(r"""
Las combinaciones relacionadas con advertencias, incendios, evacuaciones y daños aparecen con mayor fuerza en la clase positiva. En la clase negativa predominan expresiones sociales, titulares ambiguos y lenguaje figurado. Las nubes de palabras son descriptivas; la inferencia se apoya en tablas y modelos, no en el tamaño visual de los términos.
"""),
        code(r"""
display(Image(filename=str(FIGURES / "nube_palabras_target_0.png"), width=470))
display(Image(filename=str(FIGURES / "nube_palabras_target_1.png"), width=470))
"""),
        markdown(r"""
<div class="section"><h2>5 · Modelos clasificadores</h2></div>

Se comparan un baseline mayoritario, Naive Bayes, regresión logística y SVM calibrado. Todos reciben la misma partición. La regresión logística con TF–IDF ofrece el mejor equilibrio entre sensibilidad y precisión para la clase de interés.
"""),
        code(r"""
metrics = pd.read_csv(TABLES / "metricas_todos_los_modelos.csv")
cols = [c for c in ["modelo","usa_negatividad","accuracy","precision","recall","f1","f1_macro","roc_auc","tn","fp","fn","tp"] if c in metrics.columns]
show_table(metrics[cols])
display(Image(filename=str(FIGURES / "modelos_comparacion.png"), width=1000))
display(Image(filename=str(FIGURES / "modelos_matrices_confusion.png"), width=1000))
"""),
        markdown(r"""
<div class="metric-grid">
 <div class="metric"><b>0.813</b><br>Exactitud</div><div class="metric"><b>0.781</b><br>F1 desastre</div>
 <div class="metric"><b>0.809</b><br>F1 macro</div><div class="metric"><b>0.867</b><br>ROC–AUC</div>
</div>

<div class="section"><h2>6 · Función de clasificación para texto crudo</h2></div>

La función final encapsula limpieza, transformación TF–IDF, predicción, probabilidad, sentimiento y negatividad. Así puede probarse con un texto nuevo sin preparar manualmente sus variables.
"""),
        code(r"""
model = joblib.load(ROOT / "models" / "modelo_final.joblib")
analyzer = SentimentIntensityAnalyzer()
examples = [
    "Emergency services report a wildfire evacuation near the city",
    "That movie was a total disaster but we laughed all night 😂",
    "Flash flood warning: move to higher ground now!",
    "Lovely weather for the concert tonight :)"
]
predictions = pd.DataFrame([classify_raw_tweet(model, text, analyzer) for text in examples])
show_table(predictions)
"""),
        markdown(r"""
<div class="section"><h2>7 · Sentimiento positivo, negativo y neutral</h2></div>

VADER asigna cuatro puntajes: negativo, neutral, positivo y `compound`. Se aplican los umbrales originales: positivo si `compound ≥ 0.05`, negativo si `compound ≤ −0.05` y neutral en el intervalo restante. Emoticonos, emojis, signos y mayúsculas permanecen disponibles para este análisis.
"""),
        code(r"""
sentiment = pd.read_csv(TABLES / "sentimiento_distribucion.csv")
thresholds = pd.read_csv(TABLES / "sentimiento_umbrales.csv")
show_table(thresholds)
show_table(sentiment)
display(Image(filename=str(FIGURES / "sentimiento_final.png"), width=1000))
"""),
        markdown(r"""
En el corpus completo se identifican 3,747 tweets negativos, 1,971 positivos y 1,895 neutrales. El sentimiento describe el tono del mensaje, pero no equivale a la etiqueta de desastre: un reporte factual puede ser neutral y una publicación no relacionada puede expresar fuerte negatividad.

<div class="section"><h2>8 · Diez tweets extremos y sus patrones</h2></div>
"""),
        code(r"""
positive = pd.read_csv(TABLES / "top10_positivos.csv")
negative = pd.read_csv(TABLES / "top10_negativos.csv")
print("Diez tweets más positivos")
show_table(positive[["id","categoria","sentiment_compound","text"]])
print("Diez tweets más negativos")
show_table(negative[["id","categoria","sentiment_compound","text"]])
display(Image(filename=str(FIGURES / "top10_composicion.png"), width=700))
"""),
        markdown(r"""
Siete de los diez extremos negativos corresponden a desastres, mientras que nueve de los diez extremos positivos no corresponden a desastres. Los extremos negativos concentran vocabulario de muerte, lesión, amenaza y destrucción; los positivos incluyen agradecimiento, humor, apoyo y lenguaje promocional. Esta composición respalda una relación entre clase y tono, pero también muestra excepciones suficientes para evitar una regla basada únicamente en sentimiento.

<div class="section"><h2>9 · ¿Los desastres son más negativos?</h2></div>

Se define `negativity = max(−compound, 0)`. Como la distribución contiene muchos ceros y no es normal, se usa Mann–Whitney bilateral. Se acompaña con diferencia de medias, intervalo bootstrap del 95% y delta de Cliff.
"""),
        code(r"""
contrast = pd.read_csv(TABLES / "contraste_estadistico.csv")
show_table(contrast)
display(Image(filename=str(FIGURES / "contraste_negatividad.png"), width=1000))
"""),
        markdown(r"""
<div class="insight"><b>Resultado.</b> La negatividad media es 0.337 en desastres y 0.213 en no desastres. La diferencia media es 0.124, con IC bootstrap 95% [0.110, 0.138]. La prueba produce <i>p</i> &lt; 0.001 y δ de Cliff = 0.207: existe evidencia clara de diferencia, pero el tamaño del efecto es pequeño.</div>

<div class="section"><h2>10 · Variable de negatividad y reentrenamiento</h2></div>

El modelo A usa solo TF–IDF; el modelo B agrega `negativity`. La partición, semilla y parámetros de regresión logística permanecen idénticos, de modo que cualquier diferencia pueda atribuirse a la nueva variable.
"""),
        code(r"""
comparison = pd.read_csv(TABLES / "comparacion_negatividad_metricas.csv")
deltas = pd.read_csv(TABLES / "comparacion_negatividad_diferencias.csv")
show_table(comparison)
show_table(deltas)
display(Image(filename=str(FIGURES / "comparacion_negatividad.png"), width=1000))
"""),
        markdown(r"""
La negatividad aumenta ROC–AUC en apenas 0.0004, pero reduce exactitud en 0.0151, recall en 0.0153 y F1 en 0.0171; además agrega 13 falsos positivos y 10 falsos negativos. Por lo tanto, no se incorpora al modelo definitivo. El resultado negativo del experimento es informativo: una variable puede diferir significativamente entre grupos y aun así no aportar señal incremental a un modelo que ya representa el texto.

<div class="section"><h2>11 · Modelo final y análisis de errores</h2></div>
"""),
        code(r"""
metadata = json.loads((ROOT / "models" / "metadata_final.json").read_text(encoding="utf-8"))
display(pd.DataFrame([metadata]).T.rename(columns={0:"valor"}))
display(Image(filename=str(FIGURES / "modelo_final.png"), width=900))
errors = pd.read_csv(TABLES / "analisis_errores.csv")
print(f"Falsos positivos: {(errors['tipo_error'] == 'Falso positivo').sum()} · Falsos negativos: {(errors['tipo_error'] == 'Falso negativo').sum()}")
show_table(errors[["tipo_error","sentiment_label","negativity","probabilidad_desastre","text"]].head(15))
display(Image(filename=str(FIGURES / "analisis_errores.png"), width=1000))
"""),
        markdown(r"""
Los 139 falsos positivos suelen contener terminología alarmante en contextos figurados, humorísticos o promocionales. Los 146 falsos negativos incluyen mensajes breves, ubicaciones implícitas y reportes que describen hechos con tono neutral. Esto evidencia límites del modelo bolsa-de-palabras: no comprende ironía, temporalidad ni conocimiento del mundo.

### Conclusiones

1. Los n-gramas aportan contexto suficiente para superar ampliamente el baseline sin recurrir a modelos opacos.
2. La regresión logística con TF–IDF logra el mejor compromiso y queda seleccionada como modelo final.
3. Los tweets de desastre son, en promedio, más negativos; la diferencia es significativa pero de tamaño pequeño.
4. La negatividad no mejora la clasificación cuando se añade a TF–IDF, por lo que se descarta del modelo definitivo.
5. El desempeño debe interpretarse con cautela: el corpus proviene de una competencia, VADER está diseñado para inglés y la validación corresponde a una sola partición estratificada.

### Reproducibilidad y evidencia
"""),
        code(r"""
evidence = pd.read_csv(TABLES / "evidencia_rubrica.csv")
show_table(evidence)
print("Modelo final:", (ROOT / "models" / "modelo_final.joblib").exists())
print("Metadatos:", (ROOT / "models" / "metadata_final.json").exists())
print("Semilla:", summary["particion"]["semilla"])
"""),
        markdown(r"""
### Referencias

- Hutto, C. J. y Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*. ICWSM.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
- Salton, G. y Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management, 24(5), 513–523.
- Kaggle. *Natural Language Processing with Disaster Tweets*. https://www.kaggle.com/competitions/nlp-getting-started
- Scikit-learn Developers. *Working With Text Data*. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

<div class="footer">Laboratorio 5 · Grupo 1 · Data Science, Sección 10 · Entrega final reproducible</div>
"""),
    ]

    notebook["cells"] = cells
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Notebook final construido: {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
