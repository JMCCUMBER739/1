"""Generate a sample multi-chapter PDF with rich reference content for demos/tests."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


PAGES = [
    (
        "Chapter I: On the Hermetic Light",
        """
        In the beginning of this treatise we contemplate Hermes Trismegistus and the Emerald Tablet,
        wherein it is written: as above, so below. The alchemist seeks the philosopher's stone through
        nigredo, albedo, and rubedo, dissolving the prima materia by the law of solve et coagula.
        Such esoteric instruction was preserved in mystery schools and whispered among initiates.
        The astral correspondence of planetary forces guides the work of transmutation, and the
        hermetic vessel becomes a mirror of the Tree of Life and the sefirot.
        """,
    ),
    (
        "Chapter II: Theological Contours",
        """
        Theology here is not mere dogma but inquiry into the sacred. Augustine and Aquinas both
        wrestled with grace, covenant, and the Trinity. Scripture proclaims salvation through
        incarnation and resurrection, while the prophets of the Torah speak of divine justice.
        Prayer and worship orient the soul toward the holy. Comparative theology notes that
        dharma and nirvana likewise seek liberation, yet Christian eschatology frames hope in
        paradise and the final covenant of peace.
        """,
    ),
    (
        "Chapter III: The Scientific Temper",
        """
        Observation and experiment distinguish the scientific method from idle conjecture. Galileo
        turned the telescope upon celestial bodies and challenged the geocentric model with a
        heliocentric orbit. Newton described gravity, force, and acceleration; later minds probed
        the atom, the electron, and quantum phenomena. In biology, species arise by natural selection;
        anatomy and physiology map the circulation of blood. Chemistry studies compound, molecule,
        and reaction; mathematics supplies theorem, axiom, and geometry to measure the infinite.
        """,
    ),
    (
        "Chapter IV: Synthesis and Insight",
        """
        What wisdom joins the occult, the theological, and the empirical? Perhaps the divine spark
        sought by mystics is kin to the hypothesis tested by the physician. Energy and light appear
        in sacred metaphor and in optics alike. Faith without evidence risks dogma; measurement without
        meaning risks barrenness. This book therefore gathers excerpts of hermetic, theological, and
        scientific reference so that the reader may weigh each tradition with care.
        """,
    ),
]


def build_sample_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChapTitle",
        parent=styles["Heading1"],
        fontName="Times-Bold",
        fontSize=16,
        spaceAfter=14,
    )
    body_style = ParagraphStyle(
        "BodyText2",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=15,
    )
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
        title="Sample Ancient Miscellany",
    )
    story = []
    for i, (title, body) in enumerate(PAGES):
        story.append(Paragraph(title, title_style))
        cleaned = " ".join(body.split())
        story.append(Paragraph(cleaned, body_style))
        story.append(Spacer(1, 0.4 * inch))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    doc.build(story)
    return path


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "samples" / "sample_ancient_miscellany.pdf"
    build_sample_pdf(out)
    print(out)
