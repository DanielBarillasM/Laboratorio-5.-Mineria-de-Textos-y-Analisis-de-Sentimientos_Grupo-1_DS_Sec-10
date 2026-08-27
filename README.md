<div align="center">

# Laboratorio 5 · Minería de Textos y Análisis de Sentimientos

### Clasificación reproducible de tweets sobre desastres reales

![Estado](https://img.shields.io/badge/estado-avance%2075%25-0f9d91)
![Python](https://img.shields.io/badge/Python-3.11%2B-2563eb)
![Pruebas](https://img.shields.io/badge/tests-9%20passed-16a34a)
![Licencia](https://img.shields.io/badge/licencia-MIT-102a43)

**Universidad del Valle de Guatemala · Data Science · Sección 10 · Grupo 1**

</div>

Este repositorio presenta el avance del Laboratorio 5 a partir del conjunto oficial de la competencia [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started). El flujo cubre exploración, limpieza auditable, n-gramas con probabilidades, comparación preliminar de clasificadores, inferencia sobre texto crudo y sentimiento descriptivo.

## Resultado principal

La regresión logística con TF-IDF de unigramas y bigramas fue el mejor modelo preliminar sobre una partición estratificada 80/20.

| Métrica | Resultado |
|---|---:|
| Exactitud | 81.29% |
| Precisión, clase desastre | 0.785 |
| Recall, clase desastre | 0.777 |
| F1, clase desastre | 0.781 |
| F1 macro | 0.809 |
| ROC-AUC | 0.867 |

El corpus contiene 7,613 tweets: 4,342 no representan un desastre real y 3,271 sí. El baseline de clase mayoritaria logra 57.06% de exactitud y F1 igual a cero para la clase de interés, por lo que no es una solución útil.

## Entregables del avance

- [Notebook ejecutado](notebooks/Lab5_Avance_75.ipynb), con HTML/CSS incrustado, interpretación y salidas visibles.
- [Informe del avance](reports/informe_avance_75.pdf) y su [fuente LaTeX](reports/informe_avance_75.tex).
- Código modular en [`src/lab5_text`](src/lab5_text) y scripts reproducibles en [`scripts`](scripts).
- Tablas y figuras derivadas en [`outputs`](outputs).
- Pruebas automatizadas en [`tests`](tests).
- Datos originales en [`data/raw/train.csv`](data/raw/train.csv), verificados por SHA-256.

## Alcance real: 75%

| Componente | Estado |
|---|---|
| Descripción de datos y EDA | Completo |
| Limpieza explicada y auditada | Completo |
| Unigramas, bigramas y trigramas con probabilidades | Completo |
| Nubes de palabras y vocabulario distintivo | Completo |
| Baseline + Naive Bayes + regresión logística + SVM | Completo |
| Función para clasificar un tweet nuevo | Completo |
| Análisis de sentimiento | Preliminar |
| Top 10 positivo/negativo e interpretación | Pendiente para la entrega final |
| Contraste estadístico de sentimiento por clase | Pendiente para la entrega final |
| Variable de negatividad, reentrenamiento y afinación final | Pendiente para la entrega final |

La separación anterior deja deliberadamente el último 25% para la entrega definitiva y evita presentar como concluyentes resultados que aún son exploratorios.

## Reproducibilidad

Con [`uv`](https://docs.astral.sh/uv/):

```powershell
uv sync --extra test
.\.venv\Scripts\python.exe scripts\run_advance.py
.\.venv\Scripts\python.exe -m pytest -q
```

Alternativamente, con `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_advance.py
```

Clasificación desde la terminal:

```powershell
.\.venv\Scripts\python.exe scripts\predict_tweet.py "Emergency services report a wildfire evacuation"
```

La semilla es 42 y todos los modelos usan exactamente las mismas 6,090 filas de entrenamiento y 1,523 de validación. El archivo original tiene SHA-256 `61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df`.

## Estructura

```text
config/          metadatos del proyecto
data/            datos originales y derivados
notebooks/       análisis narrativo ejecutado
outputs/         figuras y tablas reproducibles
reports/         informe en LaTeX y PDF
scripts/         generación del análisis, notebook e inferencia
src/lab5_text/   limpieza, análisis, modelos y sentimiento
tests/           controles automatizados
```

## Integrantes

| Estudiante | Carné |
|---|---:|
| Jorge Gabriel Palacios Sales | 231385 |
| Pablo Daniel Barillas Moreno | 22193 |
| Roberto Emiliano Otoniel | 23968 |

## Nota metodológica

Se usan dos limpiezas: una normalización más intensa para clasificación y otra ligera para sentimiento. Esta última conserva negaciones, puntuación y emojis, señales que VADER necesita. No se almacenan credenciales ni tokens en el proyecto.

