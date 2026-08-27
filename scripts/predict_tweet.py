"""Clasifica uno o varios tweets crudos con el modelo definitivo del laboratorio.

Ejemplos
--------
    python3 scripts/predict_tweet.py "Emergency crews respond to a wildfire"
    python3 scripts/predict_tweet.py "tweet uno" "tweet dos" --formato tabla
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab5_text.final_model import classify_raw_tweet  # noqa: E402


MODEL_PATH = ROOT / "models" / "modelo_final.joblib"
METADATA_PATH = ROOT / "models" / "metadata_final.json"


def load_model():
    """Carga el modelo definitivo y su metadata, o explica cómo generarlos."""

    if not MODEL_PATH.exists():
        raise SystemExit(
            "No se encontró models/modelo_final.joblib.\n"
            "Genere los artefactos con: python3 scripts/run_final.py"
        )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    return joblib.load(MODEL_PATH), metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tweets", nargs="+", help="Uno o más textos crudos, sin preprocesar")
    parser.add_argument(
        "--formato",
        choices=("json", "tabla"),
        default="json",
        help="Formato de salida (por defecto: json)",
    )
    args = parser.parse_args()

    model, metadata = load_model()
    analyzer = SentimentIntensityAnalyzer()
    results = [classify_raw_tweet(model, tweet, analyzer) for tweet in args.tweets]

    if args.formato == "tabla":
        table = pd.DataFrame(results)[
            ["texto", "clase", "probabilidad_desastre", "sentimiento", "compound", "negatividad"]
        ]
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 60)
        print(table.to_string(index=False))
    else:
        print(json.dumps(results if len(results) > 1 else results[0], ensure_ascii=False, indent=2))

    if metadata:
        print(
            f"\nModelo: {metadata.get('modelo')} · usa negatividad: "
            f"{metadata.get('usa_negatividad')} · F1 desastre: "
            f"{metadata.get('metricas_validacion', {}).get('f1'):.6f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
