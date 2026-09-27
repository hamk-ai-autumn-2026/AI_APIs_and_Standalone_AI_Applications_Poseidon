from __future__ import annotations

from pydantic import BaseModel, Field
from openai import OpenAI

from models import ArticleSection, Paragraph, Report, Source


class ReportDraft(BaseModel):
    title: str = Field(min_length=1)
    abstract: list[Paragraph] = Field(min_length=1)
    introduction: list[Paragraph] = Field(min_length=1)
    sections: list[ArticleSection] = Field(min_length=2, max_length=6)
    conclusion: list[Paragraph] = Field(min_length=1)
    reference_ids: list[str] = Field(min_length=1)


class BaselineReference(BaseModel):
    authors: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    year: int = Field(ge=1500, le=2100)
    doi: str | None = None
    url: str | None = None


class BaselineSection(BaseModel):
    heading: str = Field(min_length=1)
    paragraphs: list[str] = Field(min_length=1)


class BaselineReport(BaseModel):
    title: str = Field(min_length=1)
    abstract: str = Field(min_length=1)
    introduction: list[str] = Field(min_length=1)
    sections: list[BaselineSection] = Field(min_length=2, max_length=6)
    conclusion: str = Field(min_length=1)
    references: list[BaselineReference] = Field(min_length=1)


def generate_grounded_report(client: OpenAI, topic: str, sources: list[Source], model: str) -> Report:
    source_context = [
        {
            "id": source.id,
            "authors": source.authors,
            "title": source.title,
            "year": source.year,
            "container_title": source.container_title,
            "summary": source.summary,
            "doi": source.doi,
            "url": str(source.url),
        }
        for source in sources
    ]
    completion = client.beta.chat.completions.parse(
        model=model,
        temperature=0.25,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a concise scientific or technical article using only the supplied sources. "
                    "Do not add facts or references from memory. Every source_ids value must be an ID "
                    "from the supplied source list; cite a source only for claims its record supports. "
                    "Use 2-4 logically ordered main sections, a clear abstract, introduction, and conclusion. "
                    "Return reference_ids for every source cited in the article and no others."
                ),
            },
            {"role": "user", "content": f"Topic: {topic}\n\nVerified search sources:\n{source_context}"},
        ],
        response_format=ReportDraft,
    )
    draft = completion.choices[0].message.parsed
    if draft is None:
        raise ValueError("The model did not return a valid structured article")

    sources_by_id = {source.id: source for source in sources}
    unknown_ids = set(draft.reference_ids) - set(sources_by_id)
    if unknown_ids:
        raise ValueError(f"The model returned unknown reference IDs: {sorted(unknown_ids)}")
    return Report(
        title=draft.title,
        abstract=draft.abstract,
        introduction=draft.introduction,
        sections=draft.sections,
        conclusion=draft.conclusion,
        references=[sources_by_id[source_id] for source_id in draft.reference_ids],
    )


def generate_unsearched_baseline(client: OpenAI, topic: str, model: str) -> BaselineReport:
    completion = client.beta.chat.completions.parse(
        model=model,
        temperature=0.25,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a concise scientific or technical article from your internal knowledge only. "
                    "Do not use web search or external tools. Include references you believe support it, "
                    "with authors, title, year, and DOI or URL when known. Never claim a reference "
                    "has been checked. Use 2-4 logically ordered main sections."
                ),
            },
            {"role": "user", "content": f"Topic: {topic}"},
        ],
        response_format=BaselineReport,
    )
    draft = completion.choices[0].message.parsed
    if draft is None:
        raise ValueError("The model did not return a valid baseline article")
    return draft


def verify_baseline(report: BaselineReport):
    from research import verify_external_reference

    return [
        verify_external_reference(
            label=f"baseline-{index}",
            authors=reference.authors,
            title=reference.title,
            year=reference.year,
            doi=reference.doi,
            url=reference.url,
        )
        for index, reference in enumerate(report.references, start=1)
    ]