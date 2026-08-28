"""Higher-level "insight" generation: keywords, notable entities, themes.

Kept dependency-light (no mandatory model downloads) so the tool works
out-of-the-box: keyword extraction is frequency-based after stopword
removal, and entity detection is a regex heuristic over capitalized
sequences ("NER-lite"). When ``OPENAI_API_KEY`` is available we additionally
ask an LLM for a richer thematic analysis.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.utils import word_tokenize_basic

_BASIC_STOPWORDS = {
    "the", "and", "of", "to", "a", "in", "that", "it", "was", "for", "on",
    "as", "with", "his", "he", "is", "at", "by", "had", "not", "but", "be",
    "her", "which", "this", "have", "from", "or", "one", "were", "an", "she",
    "their", "there", "we", "been", "has", "so", "if", "will", "would",
    "all", "when", "who", "them", "no", "more", "into", "than", "then",
    "its", "our", "you", "your", "they", "what", "some", "could", "him",
    "said", "upon", "such", "may", "these", "those", "us", "did", "do",
    "does", "am", "are", "very", "much", "any", "also", "yet", "each",
    "other", "must", "can", "thus", "even", "how", "where", "while",
}


@dataclass
class BookInsights:
    top_keywords: List[tuple] = field(default_factory=list)
    notable_entities: List[tuple] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    narrative_summary: str = ""
    sentiment_arc_description: str = ""


def _get_stopwords(iso_lang: Optional[str]) -> set:
    try:
        import nltk
        from nltk.corpus import stopwords

        lang_map = {
            "en": "english", "de": "german", "fr": "french", "es": "spanish",
            "it": "italian", "pt": "portuguese", "ru": "russian",
        }
        lang = lang_map.get(iso_lang or "en", "english")
        try:
            return set(stopwords.words(lang))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words(lang))
    except Exception:
        return _BASIC_STOPWORDS


def extract_keywords(text: str, top_n: int = 20, iso_lang: Optional[str] = None) -> List[tuple]:
    stop_words = _get_stopwords(iso_lang) | _BASIC_STOPWORDS
    tokens = [t for t in word_tokenize_basic(text) if len(t) > 3 and t not in stop_words]
    counts = Counter(tokens)
    return counts.most_common(top_n)


_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_SENTENCE_START_WORDS_TO_IGNORE = {"The", "This", "That", "These", "Those", "It", "He", "She", "They", "In", "On", "At", "As", "But", "And"}


def extract_notable_entities(text: str, top_n: int = 20) -> List[tuple]:
    """Frequency-based proper-noun ("NER-lite") extraction that needs no
    model download. Filters out common sentence-initial words.
    """
    candidates = _PROPER_NOUN_RE.findall(text)
    filtered = [c for c in candidates if c not in _SENTENCE_START_WORDS_TO_IGNORE and len(c) > 2]
    counts = Counter(filtered)
    return counts.most_common(top_n)


def _sentiment_arc_description(sentiment_by_chapter: List[Dict]) -> str:
    if not sentiment_by_chapter:
        return "Not enough chapter structure was detected to describe a sentiment arc."
    compounds = [c.get("compound", 0.0) for c in sentiment_by_chapter]
    avg = sum(compounds) / len(compounds)
    trend = compounds[-1] - compounds[0]
    tone = "predominantly positive" if avg > 0.15 else "predominantly negative" if avg < -0.15 else "emotionally neutral/balanced"
    direction = (
        "grows notably more positive toward the end"
        if trend > 0.2
        else "grows notably darker toward the end"
        if trend < -0.2
        else "stays fairly consistent in tone throughout"
    )
    return f"The text is {tone} overall (avg. sentiment {avg:+.2f}) and {direction}."


def _llm_thematic_analysis(full_text: str) -> Optional[Dict[str, object]]:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = (
            "Analyze the following (possibly OCR'd, possibly archaic) book excerpt. "
            "Return a short JSON object with keys 'themes' (list of 5-8 short theme "
            "phrases) and 'narrative_summary' (a rich 200-300 word overview of the whole "
            "work's arguments/narrative arc). Text:\n\n" + full_text[:15000]
        )
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_INSIGHT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        import json

        return json.loads(response.choices[0].message.content or "{}")
    except Exception:
        return None


def generate_insights(
    full_text: str,
    sentiment_by_chapter: List[Dict],
    iso_lang: Optional[str] = None,
    top_n_keywords: int = 20,
    use_llm: bool = True,
) -> BookInsights:
    keywords = extract_keywords(full_text, top_n=top_n_keywords, iso_lang=iso_lang)
    entities = extract_notable_entities(full_text, top_n=20)
    sentiment_desc = _sentiment_arc_description(sentiment_by_chapter)

    themes: List[str] = []
    narrative_summary = ""
    llm_data = _llm_thematic_analysis(full_text) if use_llm else None
    if llm_data:
        themes = list(llm_data.get("themes", []))[:8]
        narrative_summary = str(llm_data.get("narrative_summary", ""))
    else:
        themes = [kw for kw, _ in keywords[:8]]
        narrative_summary = (
            "Automated extractive overview (enable an OPENAI_API_KEY for a richer, "
            "abstractive narrative summary). The most frequent significant terms are: "
            + ", ".join(themes) + "."
        )

    return BookInsights(
        top_keywords=keywords,
        notable_entities=entities,
        themes=themes,
        narrative_summary=narrative_summary,
        sentiment_arc_description=sentiment_desc,
    )
