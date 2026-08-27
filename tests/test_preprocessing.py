from lab5_text.preprocessing import (
    clean_for_classification,
    clean_for_sentiment,
    surface_markers,
)


def test_cleaning_preserves_negation_and_911() -> None:
    cleaned = clean_for_classification(
        "I can't ignore #Flood911: call 911! @rescue https://example.org"
    )

    assert "not" in cleaned.split()
    assert "911" in cleaned.split()
    assert "flood" in cleaned.split()
    assert "http" not in cleaned
    assert "rescue" not in cleaned


def test_sentiment_cleaning_is_deliberately_light() -> None:
    cleaned = clean_for_sentiment("I can't believe it! 😢 #Flood @user https://x.test")

    assert "not" in cleaned
    assert "!" in cleaned
    assert "😢" in cleaned
    assert "Flood" in cleaned
    assert "@user" not in cleaned
    assert "https" not in cleaned


def test_surface_markers() -> None:
    markers = surface_markers("Help! #fire @911 https://x.test call 911 😢")

    assert markers["url_count"] == 1
    assert markers["mention_count"] == 1
    assert markers["hashtag_count"] == 1
    assert markers["exclamation_count"] == 1
    assert markers["contains_911"] == 1
    assert markers["emoji_like_count"] >= 1
