"""Clasifica un tweet nuevo con el modelo preliminar guardado."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab5_text.modeling import predict_tweet  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tweet", help="Texto crudo que se desea clasificar")
    args = parser.parse_args()

    model_path = ROOT / "models" / "modelo_preliminar.joblib"
    if not model_path.exists():
        raise SystemExit("No existe el modelo. Ejecute primero: python scripts/run_advance.py")

    model = joblib.load(model_path)
    result = predict_tweet(model, args.tweet, SentimentIntensityAnalyzer())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
