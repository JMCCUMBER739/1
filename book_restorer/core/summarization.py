"""Chapter synopsis generation.

Uses extractive summarization (LexRank, via ``sumy``) as a robust, offline,
no-API-key baseline, with an optional LLM-based abstractive summary when
``OPENAI_API_KEY`` is configured for noticeably higher quality prose.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from core.config import SUPPORTED_SUMMARY_LANGUAGES


def _extractive_summary(text: str, sentence_count: int, iso_lang: Optional[str]) -> str:
    from sumy.nlp.stemmers import Stemmer
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.utils import get_stop_words

    language = SUPPORTED_SUMMARY_LANGUAGES.get(iso_lang or "en", "english")
    try:
        tokenizer = Tokenizer(language)
    except LookupError:
        tokenizer = Tokenizer("english")
        language = "english"

    parser = PlaintextParser.from_string(text, tokenizer)
    stemmer = Stemmer(language)
    summarizer = LexRankSummarizer(stemmer)
    try:
        summarizer.stop_words = get_stop_words(language)
    except LookupError:
        pass

    sentences = summarizer(parser.document, sentence_count)
    summary = " ".join(str(s) for s in sentences).strip()
    if summary:
        return summary

    # Ultimate fallback: first few sentences verbatim.
    naive_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(naive_sentences[:sentence_count])


def _llm_summary(text: str, title: str) -> Optional[str]:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        prompt = (
            f"Write a concise, insightful synopsis (150-250 words) of the following book chapter "
            f'titled "{title}". Focus on plot/argument, key ideas, and any notable references. '
            f"Text:\n\n{text[:12000]}"
        )
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        return None


def summarize_chapter(
    text: str,
    title: str = "",
    sentence_count: int = 6,
    iso_lang: Optional[str] = None,
    prefer_llm: bool = True,
) -> str:
    text = text.strip()
    if not text:
        return "(No text available for this chapter.)"

    if prefer_llm:
        llm_result = _llm_summary(text, title)
        if llm_result:
            return llm_result

    try:
        return _extractive_summary(text, sentence_count=sentence_count, iso_lang=iso_lang)
    except Exception:
        naive_sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(naive_sentences[:sentence_count])
