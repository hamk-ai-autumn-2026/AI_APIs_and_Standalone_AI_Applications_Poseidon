import unittest

from pydantic import ValidationError

from models import ArticleSection, Paragraph, Report, Source


class ReportValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Source(
            id="source-1",
            authors=["Example, Ada"],
            title="Verified systems",
            year=2024,
            url="https://example.org/article",
            doi="10.1234/example",
            container_title="Journal of Examples",
            verification_basis="crossref",
        )

    def make_report(self, paragraph_source_id: str = "source-1") -> Report:
        return Report(
            title="A short article",
            abstract=[Paragraph(text="Summary", source_ids=[paragraph_source_id])],
            introduction=[Paragraph(text="Context")],
            sections=[
                ArticleSection(heading="Approach", paragraphs=[Paragraph(text="Method")]),
                ArticleSection(heading="Evidence", paragraphs=[Paragraph(text="Results")]),
            ],
            conclusion=[Paragraph(text="Conclusion")],
            references=[self.source],
        )

    def test_accepts_source_id_citations(self) -> None:
        self.assertEqual(self.make_report().references[0].id, "source-1")

    def test_rejects_unknown_citation_id(self) -> None:
        with self.assertRaises(ValidationError):
            self.make_report("not-in-search-results")

    def test_rejects_reference_not_cited(self) -> None:
        with self.assertRaises(ValidationError):
            Report(
                title="A short article",
                abstract=[Paragraph(text="Summary")],
                introduction=[Paragraph(text="Context")],
                sections=[
                    ArticleSection(heading="Approach", paragraphs=[Paragraph(text="Method")]),
                    ArticleSection(heading="Evidence", paragraphs=[Paragraph(text="Results")]),
                ],
                conclusion=[Paragraph(text="Conclusion")],
                references=[self.source],
            )


if __name__ == "__main__":
    unittest.main()