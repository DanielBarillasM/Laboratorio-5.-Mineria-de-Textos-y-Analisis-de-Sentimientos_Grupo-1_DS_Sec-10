"""Preprocesamiento explícito para clasificación y sentimiento de tweets."""

from __future__ import annotations

import html
import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
NEGATORS = {"no", "nor", "not", "never", "neither", "none", "cannot"}
EXTRA_STOPS = {"rt", "amp", "https", "http", "co", "im", "u", "ur"}
STOPWORDS = (set(ENGLISH_STOP_WORDS) | EXTRA_STOPS) - NEGATORS


def _expand_contractions(text: str) -> str:
    replacements = (
        (r"\bwon['’]t\b", "will not"),
        (r"\bcan['’]t\b", "can not"),
        (r"n['’]t\b", " not"),
        (r"['’]re\b", " are"),
        (r"['’]ve\b", " have"),
        (r"['’]ll\b", " will"),
        (r"['’]d\b", " would"),
        (r"['’]m\b", " am"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_for_classification(value: object) -> str:
    """Normaliza un tweet sin destruir negaciones ni el token 911.

    Las URL y menciones se eliminan; el símbolo # se retira conservando la
    palabra. Los emoticones se omiten en esta representación, pero el análisis
    de sentimiento utiliza una versión separada que sí los preserva.
    """

    if not isinstance(value, str):
        return ""
    text = html.unescape(value).lower()
    text = _expand_contractions(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = re.sub(r"\b911\b", " nineoneone ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.encode("ascii", "ignore").decode("ascii")
    tokens = [token for token in text.split() if token and token not in STOPWORDS]
    tokens = ["911" if token == "nineoneone" else token for token in tokens]
    return " ".join(tokens)


def clean_for_sentiment(value: object) -> str:
    """Limpieza ligera que conserva puntuación, negaciones y emoticones."""

    if not isinstance(value, str):
        return ""
    text = html.unescape(value)
    text = _expand_contractions(text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", "")
    return WHITESPACE_RE.sub(" ", text).strip()


def surface_markers(value: object) -> dict[str, int]:
    """Cuenta marcas de superficie antes de limpiar el texto."""

    text = value if isinstance(value, str) else ""
    return {
        "url_count": len(URL_RE.findall(text)),
        "mention_count": len(MENTION_RE.findall(text)),
        "hashtag_count": text.count("#"),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "digit_count": sum(char.isdigit() for char in text),
        "contains_911": int(bool(re.search(r"\b911\b", text))),
        "emoji_like_count": sum(ord(char) > 127 for char in text),
    }
