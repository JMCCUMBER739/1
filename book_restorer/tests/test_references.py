from core.references import find_all_references, find_references_in_text


def test_find_references_in_text_finds_esoteric_terms_with_excerpt():
    text = (
        "The old manuscript spoke at length of alchemy and the philosopher's stone. "
        "Many scholars dismissed it as pure occultism, yet its influence endured."
    )
    refs = find_references_in_text(text, category="esoteric", page=3)
    terms = {r.term.lower() for r in refs}
    assert "alchemy" in terms
    assert "philosopher's stone" in terms
    assert all(r.page == 3 for r in refs)
    assert all(len(r.excerpt) > 0 for r in refs)


def test_find_references_in_text_finds_theological_terms():
    text = "The prophet spoke of salvation, resurrection, and the covenant with god."
    refs = find_references_in_text(text, category="theological", page=1)
    terms = {r.term.lower() for r in refs}
    assert "salvation" in terms
    assert "resurrection" in terms


def test_find_references_in_text_finds_scientific_terms():
    text = "He proposed a hypothesis about atomic structure and tested it through experiment."
    refs = find_references_in_text(text, category="scientific", page=2)
    terms = {r.term.lower() for r in refs}
    assert "hypothesis" in terms
    assert "experiment" in terms


def test_find_all_references_across_pages():
    pages = {
        0: "Alchemy and the tarot were discussed at length.",
        1: "The prophet delivered a sermon about salvation.",
    }
    result = find_all_references(pages, categories=["esoteric", "theological"])
    assert len(result["esoteric"]) >= 1
    assert len(result["theological"]) >= 1
    assert result["esoteric"][0].page == 0
    assert result["theological"][0].page == 1


def test_longer_phrases_take_priority_over_substrings():
    text = "They discussed sacred geometry at the gathering."
    refs = find_references_in_text(text, category="esoteric", page=0)
    terms = [r.term.lower() for r in refs]
    assert "sacred geometry" in terms
    # "sacred" alone should not double-match inside "sacred geometry".
    assert terms.count("sacred geometry") == 1
