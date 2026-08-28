from core.chapters import PageText, split_into_chapters


def test_split_into_chapters_detects_headings():
    pages = [
        PageText(0, "CHAPTER I\n\n" + ("Once upon a time in a land far away. " * 60)),
        PageText(1, ("The story continued for quite a while. " * 60)),
        PageText(2, "CHAPTER II\n\n" + ("A new adventure began here. " * 60)),
        PageText(3, ("It kept going and going. " * 60)),
    ]
    chapters = split_into_chapters(pages, min_chapter_chars=50)
    assert len(chapters) == 2
    assert chapters[0].number == 1
    assert chapters[1].number == 2
    assert "adventure" in chapters[1].text.lower()


def test_split_into_chapters_falls_back_to_single_chapter_without_headings():
    pages = [PageText(0, "Just some plain text with no headings at all, spanning a page.")]
    chapters = split_into_chapters(pages)
    assert len(chapters) == 1
    assert chapters[0].title == "Full Text"


def test_short_false_positive_headings_get_merged():
    pages = [
        PageText(0, "CHAPTER I\n\n" + ("Real content here. " * 100)),
        PageText(1, "YES\n\n" + ("More real content. " * 5)),
    ]
    chapters = split_into_chapters(pages, min_chapter_chars=200)
    # "YES" alone should not create a spurious tiny chapter.
    assert len(chapters) == 1
