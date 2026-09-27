from __future__ import annotations

from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph as PdfParagraph, SimpleDocTemplate, Spacer

from models import Paragraph, Report, Source


def _family(name: str) -> str:
    return name.split(",", 1)[0].strip() if "," in name else name.split()[-1]


def _in_text_citation(source: Source) -> str:
    if len(source.authors) == 1:
        authors = _family(source.authors[0])
    elif len(source.authors) == 2:
        authors = f"{_family(source.authors[0])} & {_family(source.authors[1])}"
    else:
        authors = f"{_family(source.authors[0])} et al."
    return f"{authors}, {source.year}"


def _apa_author(name: str) -> str:
    if "," in name:
        family, given = (part.strip() for part in name.split(",", 1))
        initials = " ".join(f"{part[0].upper()}." for part in given.split() if part)
        return f"{family}, {initials}" if initials else family
    parts = name.split()
    if len(parts) == 1:
        return name
    initials = " ".join(f"{part[0].upper()}." for part in parts[:-1])
    return f"{parts[-1]}, {initials}"


def _apa_authors(authors: list[str]) -> str:
    formatted = [_apa_author(author) for author in authors[:20]]
    if len(authors) > 20:
        return ", ".join(formatted[:19]) + ", ... " + _apa_author(authors[-1])
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"


def _paragraph_text(paragraph: Paragraph, references: dict[str, Source]) -> str:
    text = escape(paragraph.text)
    citations = [
        _in_text_citation(references[source_id])
        for source_id in paragraph.source_ids
        if source_id in references
    ]
    if citations:
        text += f" ({escape('; '.join(citations))})"
    return text


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d8e0df"))
    canvas.line(0.75 * inch, 0.62 * inch, 7.75 * inch, 0.62 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#546262"))
    canvas.drawString(0.75 * inch, 0.42 * inch, "Grounded Article Studio")
    canvas.drawRightString(7.75 * inch, 0.42 * inch, str(document.page))
    canvas.restoreState()


def render_pdf(report: Report) -> bytes:
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.85 * inch,
        title=report.title,
        author="Grounded Article Studio",
    )
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ArticleTitle",
        parent=base["Title"],
        fontName="Times-Bold",
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#183d3b"),
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "ArticleHeading",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#235b56"),
        spaceBefore=13,
        spaceAfter=7,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "ArticleBody",
        parent=base["BodyText"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#202928"),
        spaceAfter=8,
        alignment=4,
    )
    abstract_style = ParagraphStyle(
        "Abstract",
        parent=body_style,
        leftIndent=16,
        rightIndent=16,
        fontName="Times-Italic",
    )
    reference_style = ParagraphStyle(
        "Reference",
        parent=body_style,
        leftIndent=20,
        firstLineIndent=-20,
        fontSize=9.3,
        leading=13,
        spaceAfter=6,
    )

    references = {source.id: source for source in report.references}
    story = [PdfParagraph(escape(report.title), title_style), HRFlowable(width="100%", color="#9bb8b3")]
    story.extend((Spacer(1, 10), PdfParagraph("Abstract", heading_style)))
    story.extend(PdfParagraph(_paragraph_text(item, references), abstract_style) for item in report.abstract)
    story.append(PdfParagraph("Introduction", heading_style))
    story.extend(PdfParagraph(_paragraph_text(item, references), body_style) for item in report.introduction)

    for section in report.sections:
        story.append(PdfParagraph(escape(section.heading), heading_style))
        story.extend(PdfParagraph(_paragraph_text(item, references), body_style) for item in section.paragraphs)

    story.append(PdfParagraph("Conclusion", heading_style))
    story.extend(PdfParagraph(_paragraph_text(item, references), body_style) for item in report.conclusion)
    story.append(PdfParagraph("References", heading_style))
    for source in report.references:
        link = f"https://doi.org/{source.doi}" if source.doi else str(source.url)
        reference = (
            f"{escape(_apa_authors(source.authors))} ({source.year}). "
            f"{escape(source.title)}."
        )
        if source.container_title:
            reference += f" <i>{escape(source.container_title)}</i>."
        reference += f' <link href="{escape(link, quote=True)}" color="#176b65">{escape(link)}</link>'
        story.append(PdfParagraph(reference, reference_style))

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output.getvalue()