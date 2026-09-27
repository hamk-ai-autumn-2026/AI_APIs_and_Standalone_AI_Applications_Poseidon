from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    year: int = Field(ge=1500, le=2100)
    url: AnyHttpUrl
    doi: str | None = None
    container_title: str | None = None
    summary: str | None = None
    verification_basis: Literal["crossref", "publisher_metadata", "search_result_crossref"]
    search_result_title: str | None = None


class Paragraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class ArticleSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    heading: str = Field(min_length=1)
    paragraphs: list[Paragraph] = Field(min_length=1)


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    abstract: list[Paragraph] = Field(min_length=1)
    introduction: list[Paragraph] = Field(min_length=1)
    sections: list[ArticleSection] = Field(min_length=2, max_length=6)
    conclusion: list[Paragraph] = Field(min_length=1)
    references: list[Source] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_citations(self) -> Report:
        reference_ids = [source.id for source in self.references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("Reference IDs must be unique")

        cited_ids = {
            source_id
            for paragraph in self.all_paragraphs()
            for source_id in paragraph.source_ids
        }
        unknown_ids = cited_ids - set(reference_ids)
        unused_ids = set(reference_ids) - cited_ids
        if unknown_ids:
            raise ValueError(f"Citations reference unknown sources: {sorted(unknown_ids)}")
        if unused_ids:
            raise ValueError(f"References are not cited in the article: {sorted(unused_ids)}")
        return self

    def all_paragraphs(self) -> list[Paragraph]:
        paragraphs = [*self.abstract, *self.introduction, *self.conclusion]
        for section in self.sections:
            paragraphs.extend(section.paragraphs)
        return paragraphs