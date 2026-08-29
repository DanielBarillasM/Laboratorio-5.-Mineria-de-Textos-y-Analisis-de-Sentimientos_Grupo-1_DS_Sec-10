<div align="center">

# Laboratorio 5 · Minería de Textos y Análisis de Sentimientos

### Clasificación reproducible de tweets sobre desastres reales

![Estado](https://img.shields.io/badge/estado-entrega%20final-0f9d91)
![Python](https://img.shields.io/badge/Python-3.11%2B-2563eb)
![Pruebas](https://img.shields.io/badge/tests-71%20passed-16a34a)
![Licencia](https://img.shields.io/badge/licencia-MIT-102a43)

**Universidad del Valle de Guatemala · Data Science · Sección 10 · Grupo 1**

[Repositorio en GitHub](https://github.com/DanielBarillasM/Laboratorio-5.-Mineria-de-Textos-y-Analisis-de-Sentimientos_Grupo-1_DS_Sec-10)

</div>

## Descripción

Este repositorio contiene la entrega final del Laboratorio 5. Se utiliza el conjunto oficial de la competencia [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) para estudiar el lenguaje de 7,613 tweets y construir un clasificador capaz de distinguir desastres reales de usos figurados o noticias no relacionadas.

El flujo incluye validación de datos, limpieza auditable, frecuencias y n-gramas por clase, análisis exploratorio, comparación de clasificadores, análisis de sentimiento con VADER, contraste estadístico de negatividad, reentrenamiento con una variable adicional, selección del modelo final y análisis de errores.

## Resultado principal

El modelo definitivo es una regresión logística con TF–IDF de unigramas y bigramas. Sobre una partición estratificada 80/20 obtiene:

| Métrica | Resultado |
|---|---:|
| Exactitud | 81.29% |
| Precisión, clase desastre | 0.785 |
| Recall, clase desastre | 0.777 |
| F1, clase desastre | 0.781 |
| F1 macro | 0.809 |
| ROC-AUC | 0.867 |

Los tweets de desastres presentan mayor negatividad media que los demás: 0.337 frente a 0.213. La diferencia es estadísticamente significativa (Mann–Whitney, `p < 0.001`), aunque su tamaño de efecto es pequeño (`δ de Cliff = 0.207`). Incorporar la negatividad como predictor redujo el F1 de 0.781 a 0.764, por lo que el modelo final conserva únicamente la representación textual.

## Entregables

- [Notebook final ejecutado](notebooks/Lab5_Completo.ipynb), con narrativa, HTML/CSS y salidas visibles.
- [Informe final en PDF](reports/informe_final.pdf) y [fuente LaTeX](reports/informe_final.tex).
- [Ficha del repositorio](ficha_repositorio/Ficha_Repositorio_Laboratorio_5.docx), ubicada en una carpeta independiente.
- Código modular en [`src/lab5_text`](src/lab5_text) y scripts reproducibles en [`scripts`](scripts).
- Tablas y figuras derivadas en [`outputs`](outputs).
- Modelo y metadatos finales en [`models`](models).
- Pruebas automatizadas en [`tests`](tests).

## Reproducibilidad

Con [`uv`](https://docs.astral.sh/uv/):

```powershell
uv sync --extra test
.\.venv\Scripts\python.exe scripts\run_advance.py
.\.venv\Scripts\python.exe scripts\run_final.py
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe -m pytest -q
```

Alternativamente, con `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_advance.py
.\.venv\Scripts\python.exe scripts\run_final.py
.\.venv\Scripts\python.exe scripts\build_notebook.py
.\.venv\Scripts\python.exe -m pytest -q
```

Para volver a ejecutar el notebook:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks\Lab5_Completo.ipynb --ExecutePreprocessor.timeout=600
```

### Clasificar un tweet nuevo

```powershell
.\.venv\Scripts\python.exe scripts\predict_tweet.py "Emergency services report a wildfire evacuation"
```

La función recibe texto crudo y devuelve clase, etiqueta, probabilidad de desastre, sentimiento y negatividad. La semilla del experimento es 42; la división contiene 6,090 observaciones de entrenamiento y 1,523 de validación.

## Estructura

```text
config/             metadatos del proyecto
data/               datos originales y guía de procedencia
ficha_repositorio/  ficha DOCX para presentar el enlace
models/             modelo final y metadatos de reproducción
notebooks/          análisis narrativo ejecutado
outputs/            figuras y tablas reproducibles
reports/            informe final en LaTeX y PDF
scripts/            flujos de análisis, notebook e inferencia
src/lab5_text/      limpieza, análisis, modelos y sentimiento
tests/              controles automatizados
```

## Integrantes

| Estudiante | Carné |
|---|---:|
| Jorge Gabriel Palacios Sales | 231385 |
| Pablo Daniel Barillas Moreno | 22193 |
| Roberto Emiliano Otoniel | 23968 |

## Decisiones metodológicas

- Se mantiene una normalización intensa para el clasificador y una limpieza ligera para VADER, pues este último necesita negaciones, puntuación y emoticonos.
- La lista de palabras vacías se audita porque algunas palabras temáticas, como `fire`, pueden contener señal útil.
- Todas las comparaciones usan la misma partición estratificada y la misma semilla.
- El archivo `test.csv` de Kaggle se conserva como referencia, pero el laboratorio se evalúa con `train.csv`, único archivo que contiene la etiqueta `target`.
- No se almacenan credenciales, contraseñas ni tokens en el proyecto.

## Referencias principales

- Hutto, C. J. y Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text*.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
- Kaggle. *Natural Language Processing with Disaster Tweets*.
