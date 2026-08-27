# Datos

`data/raw/train.csv` corresponde al conjunto etiquetado de la competencia de Kaggle **Natural Language Processing with Disaster Tweets**.

## Validación del archivo

- Filas: 7,613.
- Columnas: `id`, `keyword`, `location`, `text`, `target`.
- No desastre (`target=0`): 4,342.
- Desastre real (`target=1`): 3,271.
- SHA-256: `61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df`.

Kaggle exige autenticación y aceptación de las reglas para la descarga oficial. La copia local utilizada se verificó contra la estructura y conteos conocidos de la competencia. Fuente pública de recuperación:

```text
https://github.com/tarunannapareddy/Natural-Language-Processing-with-Disaster-Tweets
```

Los archivos de `data/processed/` se generan mediante `scripts/run_advance.py` y no se versionan.
