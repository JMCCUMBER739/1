"""Reference mining: esoteric, theological and scientific passages.

Curated multi-word lexicons are scanned against every sentence of every
chapter. Each hit records the matched terms, the verbatim excerpt, and
its chapter/page location, so findings can always be traced back to the
source. Passages that blend two or more traditions are flagged as
syncretic — often the most interesting material in antiquarian books.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .chapters import Chapter
from .textutils import split_sentences

ESOTERIC_TERMS = [
    "alchemy",
    "alchemist",
    "alchemical",
    "hermetic",
    "hermeticism",
    "kabbalah",
    "cabala",
    "qabalah",
    "occult",
    "occultism",
    "arcane",
    "arcanum",
    "esoteric",
    "gnosis",
    "gnostic",
    "gnosticism",
    "mysticism",
    "mystic",
    "mystical",
    "astrology",
    "astrologer",
    "horoscope",
    "zodiac",
    "talisman",
    "amulet",
    "sigil",
    "grimoire",
    "necromancy",
    "divination",
    "scrying",
    "augury",
    "tarot",
    "rosicrucian",
    "freemason",
    "masonic",
    "theosophy",
    "theosophical",
    "philosopher's stone",
    "philosophers stone",
    "elixir of life",
    "prima materia",
    "magnum opus",
    "transmutation",
    "emerald tablet",
    "as above so below",
    "sacred geometry",
    "numerology",
    "clairvoyance",
    "seance",
    "spiritualism",
    "astral",
    "etheric",
    "invocation",
    "evocation",
    "pentagram",
    "hexagram",
    "rune",
    "runic",
    "initiation",
    "adept",
    "hierophant",
    "magus",
    "sorcery",
    "sorcerer",
    "witchcraft",
    "enchantment",
    "incantation",
    "conjuration",
    "spell",
    "third eye",
    "aura",
    "chakra",
    "occulted",
    "hidden wisdom",
    "secret doctrine",
    "mysteries",
    "oracle",
    "prophecy",
    "prophetic",
    "apparition",
    "spectre",
    "phantom",
]

THEOLOGICAL_TERMS = [
    "god",
    "gods",
    "goddess",
    "divine",
    "divinity",
    "deity",
    "theology",
    "theological",
    "scripture",
    "scriptural",
    "gospel",
    "biblical",
    "bible",
    "testament",
    "psalm",
    "prophet",
    "apostle",
    "messiah",
    "christ",
    "jesus",
    "holy spirit",
    "holy ghost",
    "trinity",
    "salvation",
    "sin",
    "redemption",
    "resurrection",
    "heaven",
    "hell",
    "purgatory",
    "paradise",
    "angel",
    "angelic",
    "archangel",
    "demon",
    "demonic",
    "devil",
    "satan",
    "lucifer",
    "soul",
    "spirit",
    "eternal life",
    "immortality",
    "creation",
    "creator",
    "providence",
    "grace",
    "faith",
    "prayer",
    "worship",
    "sacrament",
    "baptism",
    "eucharist",
    "communion",
    "church",
    "temple",
    "synagogue",
    "mosque",
    "priest",
    "priesthood",
    "bishop",
    "monk",
    "monastery",
    "nun",
    "clergy",
    "pope",
    "papal",
    "covenant",
    "commandment",
    "revelation",
    "apocalypse",
    "judgment day",
    "last judgment",
    "genesis",
    "exodus",
    "torah",
    "talmud",
    "koran",
    "quran",
    "vedas",
    "upanishads",
    "dharma",
    "karma",
    "nirvana",
    "reincarnation",
    "buddha",
    "brahman",
    "allah",
    "jehovah",
    "yahweh",
    "atonement",
    "blasphemy",
    "heresy",
    "heretic",
    "martyr",
    "saint",
    "sainthood",
    "miracle",
    "miraculous",
    "blessed",
    "sanctified",
    "consecrated",
    "hallowed",
    "almighty",
]

SCIENTIFIC_TERMS = [
    "science",
    "scientific",
    "experiment",
    "experimental",
    "hypothesis",
    "theory",
    "observation",
    "phenomenon",
    "phenomena",
    "physics",
    "chemistry",
    "chemical",
    "biology",
    "anatomy",
    "physiology",
    "botany",
    "zoology",
    "geology",
    "astronomy",
    "astronomer",
    "telescope",
    "microscope",
    "mathematics",
    "mathematical",
    "geometry",
    "algebra",
    "calculus",
    "equation",
    "electricity",
    "electrical",
    "magnetism",
    "magnetic",
    "gravity",
    "gravitation",
    "velocity",
    "momentum",
    "energy",
    "matter",
    "atom",
    "atomic",
    "molecule",
    "element",
    "compound",
    "combustion",
    "oxygen",
    "hydrogen",
    "nitrogen",
    "carbon",
    "mercury",
    "sulphur",
    "sulfur",
    "phlogiston",
    "ether",
    "aether",
    "corpuscle",
    "specimen",
    "dissection",
    "circulation",
    "nervous system",
    "organism",
    "species",
    "evolution",
    "natural selection",
    "naturalist",
    "natural philosophy",
    "natural philosopher",
    "empirical",
    "empiricism",
    "rational",
    "logic",
    "deduction",
    "induction",
    "measurement",
    "instrument",
    "apparatus",
    "laboratory",
    "distillation",
    "crucible",
    "furnace",
    "solution",
    "precipitate",
    "crystallization",
    "optics",
    "refraction",
    "spectrum",
    "lens",
    "orbit",
    "planet",
    "planetary",
    "comet",
    "eclipse",
    "celestial",
    "terrestrial",
    "longitude",
    "latitude",
    "medicine",
    "medical",
    "physician",
    "surgeon",
    "remedy",
    "anatomy",
    "diagnosis",
    "epidemic",
    "contagion",
    "inoculation",
    "vaccination",
]

CATEGORY_LEXICONS: dict[str, list[str]] = {
    "esoteric": ESOTERIC_TERMS,
    "theological": THEOLOGICAL_TERMS,
    "scientific": SCIENTIFIC_TERMS,
}

# Generic terms that shouldn't dominate results get half weight.
_LOW_WEIGHT = {
    "spell",
    "spirit",
    "soul",
    "theory",
    "matter",
    "energy",
    "rational",
    "logic",
    "element",
    "solution",
    "god",
    "faith",
    "mysteries",
    "instrument",
    "species",
    "orbit",
}


@dataclass
class ReferenceHit:
    category: str
    terms: list[str]
    excerpt: str
    chapter_number: int
    chapter_title: str
    start_page: int
    end_page: int
    score: float

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "terms": self.terms,
            "excerpt": self.excerpt,
            "chapter": self.chapter_number,
            "chapter_title": self.chapter_title,
            "pages": f"{self.start_page}-{self.end_page}",
            "score": round(self.score, 2),
        }


@dataclass
class ReferenceReport:
    hits: dict[str, list[ReferenceHit]] = field(default_factory=dict)
    syncretic: list[ReferenceHit] = field(default_factory=list)
    term_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    chapter_density: dict[str, dict[int, float]] = field(default_factory=dict)

    def total(self, category: str) -> int:
        return len(self.hits.get(category, []))


def _compile_lexicon(terms: list[str]) -> re.Pattern:
    escaped = sorted((re.escape(t) for t in set(terms)), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


_PATTERNS = {cat: _compile_lexicon(terms) for cat, terms in CATEGORY_LEXICONS.items()}


def _trim_excerpt(sentence: str, limit: int = 420) -> str:
    sentence = sentence.strip()
    if len(sentence) <= limit:
        return sentence
    return sentence[:limit].rsplit(" ", 1)[0] + " …"


def scan_references(chapters: list[Chapter], max_hits_per_category: int = 250) -> ReferenceReport:
    report = ReferenceReport(
        hits={cat: [] for cat in CATEGORY_LEXICONS},
        term_counts={cat: {} for cat in CATEGORY_LEXICONS},
        chapter_density={cat: {} for cat in CATEGORY_LEXICONS},
    )

    for chapter in chapters:
        sentences = split_sentences(chapter.text)
        chapter_words = max(1, len(chapter.text.split()))
        per_chapter_counts = {cat: 0 for cat in CATEGORY_LEXICONS}

        for sentence in sentences:
            matched_categories: dict[str, list[str]] = {}
            for category, pattern in _PATTERNS.items():
                found = pattern.findall(sentence)
                if not found:
                    continue
                terms = sorted({t.lower() for t in found})
                matched_categories[category] = terms
                per_chapter_counts[category] += len(found)
                for term in terms:
                    counts = report.term_counts[category]
                    counts[term] = counts.get(term, 0) + 1

            for category, terms in matched_categories.items():
                score = sum(0.5 if t in _LOW_WEIGHT else 1.0 for t in terms)
                hit = ReferenceHit(
                    category=category,
                    terms=terms,
                    excerpt=_trim_excerpt(sentence),
                    chapter_number=chapter.number,
                    chapter_title=chapter.title,
                    start_page=chapter.start_page,
                    end_page=chapter.end_page,
                    score=score,
                )
                report.hits[category].append(hit)

            if len(matched_categories) >= 2:
                all_terms = sorted({t for ts in matched_categories.values() for t in ts})
                report.syncretic.append(
                    ReferenceHit(
                        category="+".join(sorted(matched_categories)),
                        terms=all_terms,
                        excerpt=_trim_excerpt(sentence),
                        chapter_number=chapter.number,
                        chapter_title=chapter.title,
                        start_page=chapter.start_page,
                        end_page=chapter.end_page,
                        score=float(len(all_terms)),
                    )
                )

        for category, count in per_chapter_counts.items():
            report.chapter_density[category][chapter.number] = round(
                1000.0 * count / chapter_words, 2
            )  # hits per 1k words

    for category in report.hits:
        report.hits[category].sort(key=lambda h: h.score, reverse=True)
        report.hits[category] = report.hits[category][:max_hits_per_category]
    report.syncretic.sort(key=lambda h: h.score, reverse=True)
    report.syncretic = report.syncretic[:100]
    return report
