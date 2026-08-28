from core import analytics
from core.chapters import Chapter


def test_compute_text_statistics_basic():
    text = "This is a simple sentence. Here is another one for good measure."
    stats = analytics.compute_text_statistics(text)
    assert stats.word_count > 0
    assert stats.sentence_count == 2
    assert 0 < stats.lexical_diversity <= 1
    assert stats.estimated_reading_minutes > 0


def test_compute_text_statistics_empty():
    stats = analytics.compute_text_statistics("")
    assert stats.word_count == 0


def test_compute_word_frequency_excludes_short_words():
    text = "the cat sat on the mat with a hat and a bat"
    freq = analytics.compute_word_frequency(text, top_n=10)
    words = [w for w, _ in freq]
    assert "cat" in words
    assert "on" not in words  # length <= 2 filtered out ("on" has len 2)


def test_compute_sentiment_polarity_direction():
    positive = analytics.compute_sentiment("This is wonderful, joyful, and beautiful news!")
    negative = analytics.compute_sentiment("This is terrible, sad, and awful news.")
    assert positive["compound"] > negative["compound"]


def test_compute_sentiment_by_chapter():
    chapters = [
        Chapter(number=1, title="A", text="A wonderful and joyful beginning.", start_page=0, end_page=0),
        Chapter(number=2, title="B", text="A terrible and awful ending.", start_page=1, end_page=1),
    ]
    results = analytics.compute_sentiment_by_chapter(chapters)
    assert len(results) == 2
    assert results[0]["compound"] > results[1]["compound"]


def test_compute_reference_stats_counts_terms():
    class FakeRef:
        def __init__(self, term):
            self.term = term

    refs_by_cat = {"esoteric": [FakeRef("alchemy"), FakeRef("alchemy"), FakeRef("tarot")]}
    stats = analytics.compute_reference_stats(refs_by_cat)
    assert stats["esoteric"]["total_matches"] == 3
    assert stats["esoteric"]["unique_terms"] == 2
    assert stats["esoteric"]["top_terms"][0] == ("alchemy", 2)


def test_compute_image_quality_stats():
    metrics = [
        {"skew_angle_degrees": 2.0, "noise_before": 100.0, "noise_after": 50.0},
        {"skew_angle_degrees": -1.0, "noise_before": 80.0, "noise_after": 40.0},
    ]
    result = analytics.compute_image_quality_stats(metrics)
    assert result.pages_processed == 2
    assert result.avg_skew_correction_degrees == 1.5
    assert result.avg_noise_reduction_pct == 50.0
