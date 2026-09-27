from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from models import Source


USER_AGENT = "GroundedArticle/1.0 (academic reference verification)"
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class Verification:
    label: str
    author_matches: bool
    title_matches: bool
    year_matches: bool
    locator_matches: bool
    detail: str

    @property
    def genuine(self) -> bool:
        return all((self.author_matches, self.title_matches, self.year_matches, self.locator_matches))


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _same_title(left: str, right: str) -> bool:
    left_normalized = _normalize(left)
    right_normalized = _normalize(right)
    shorter, longer = sorted((left_normalized, right_normalized), key=len)
    if len(shorter) >= 20 and longer.startswith(shorter):
        return True
    return SequenceMatcher(None, left_normalized, right_normalized).ratio() >= 0.88


def _year(value: str | None) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", value or "")
    return int(match.group()) if match else None


def _doi(value: str | None) -> str | None:
    if not value:
        return None
    match = DOI_PATTERN.search(value.strip())
    return match.group().rstrip(".,;)") if match else None


def _meta(soup: BeautifulSoup, *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        pattern = re.compile(f"^{re.escape(name)}$", re.I)
        tags = soup.find_all("meta", attrs={"name": pattern})
        tags += soup.find_all("meta", attrs={"property": pattern})
        for tag in tags:
            content = tag.get("content")
            if isinstance(content, str) and content.strip():
                values.append(content.strip())
    return values


def _jsonld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if any(key in value for key in ("headline", "datePublished", "author")):
                records.append(value)
            for key in ("@graph", "mainEntity"):
                if key in value:
                    visit(value[key])

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            visit(json.loads(script.string or script.get_text()))
        except (json.JSONDecodeError, TypeError):
            continue
    return records


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        if value.get("family"):
            given = str(value.get("given", "")).strip()
            family = str(value["family"]).strip()
            return [f"{family}, {given}" if given else family]
        return [str(value["name"]).strip()] if value.get("name") else []
    if isinstance(value, list):
        return [name for item in value for name in _names(item)]
    return []


def _page_metadata(html: str, url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    records = _jsonld(soup)
    title = next(iter(_meta(soup, "citation_title", "dc.title", "og:title")), "")
    if not title and records:
        title = str(records[0].get("headline") or records[0].get("name") or "")
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)

    authors = _meta(soup, "citation_author", "dc.creator", "author")
    if not authors and records:
        authors = [author for record in records for author in _names(record.get("author"))]
    dates = _meta(soup, "citation_publication_date", "citation_date", "dc.date", "article:published_time")
    if not dates and records:
        dates = [str(record.get("datePublished", "")) for record in records]

    doi_values = _meta(soup, "citation_doi", "dc.identifier")
    doi = next((_doi(value) for value in doi_values if _doi(value)), None)
    if not doi:
        for link in soup.find_all("a", href=True):
            doi = _doi(str(link["href"]))
            if doi:
                break
    container = next(iter(_meta(soup, "citation_journal_title", "dc.source", "og:site_name")), "")
    summary = next(iter(_meta(soup, "citation_abstract", "dc.description", "description", "og:description")), "")
    return {
        "title": title,
        "authors": list(dict.fromkeys(name for name in authors if name)),
        "year": next((found for value in dates if (found := _year(value))), None),
        "doi": doi,
        "container_title": container,
        "summary": summary[:1400],
        "url": url,
    }


def _fetch_page(url: str) -> tuple[str, str]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15, allow_redirects=True)
    response.raise_for_status()
    return response.text, response.url


def _crossref(doi: str) -> dict[str, Any] | None:
    response = requests.get(
        f"https://api.crossref.org/works/{quote(doi, safe='/')}",
        headers={"User-Agent": USER_AGENT},
        timeout=12,
    )
    if response.status_code != 200:
        return None
    return response.json().get("message")


def _crossref_search(title: str) -> dict[str, Any] | None:
    response = requests.get(
        "https://api.crossref.org/works",
        params={"query.title": title, "rows": 5},
        headers={"User-Agent": USER_AGENT},
        timeout=12,
    )
    response.raise_for_status()
    items = response.json().get("message", {}).get("items", [])
    matches = [item for item in items if _same_title(title, _crossref_title(item))]
    return max(matches, key=lambda item: SequenceMatcher(None, _normalize(title), _normalize(_crossref_title(item))).ratio(), default=None)


def _crossref_authors(record: dict[str, Any]) -> list[str]:
    names = []
    for author in record.get("author", []):
        family = str(author.get("family", "")).strip()
        given = str(author.get("given", "")).strip()
        if family:
            names.append(f"{family}, {given}" if given else family)
    return names


def _crossref_year(record: dict[str, Any]) -> int | None:
    for field in ("published-print", "published-online", "published", "created"):
        parts = record.get(field, {}).get("date-parts", [[]])
        if parts and parts[0] and isinstance(parts[0][0], int):
            return parts[0][0]
    return None


def _crossref_title(record: dict[str, Any]) -> str:
    titles = record.get("title") or []
    return str(titles[0]).strip() if titles else ""


def _surname(name: str) -> str:
    return _normalize(name.split(",", 1)[0] if "," in name else name.split()[-1])


def _authors_agree(page_authors: list[str], record_authors: list[str]) -> bool:
    if not page_authors:
        return True
    known = {_surname(author) for author in record_authors}
    return all(_surname(author) in known for author in page_authors)


def _stable_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def _source_from_result(result: dict[str, Any]) -> Source | None:
    result_url = str(result.get("href") or result.get("url") or "")
    result_title = str(result.get("title") or "").strip()
    if urlparse(result_url).scheme not in {"http", "https"}:
        return None
    try:
        html, final_url = _fetch_page(result_url)
    except requests.RequestException:
        return _source_from_search_result(result, result_url, result_title)
    page = _page_metadata(html, final_url)
    doi = page["doi"] or _doi(result_url)

    if doi:
        record = _crossref(doi)
        if not record:
            return None
        title = _crossref_title(record)
        authors = _crossref_authors(record)
        year = _crossref_year(record)
        if not title or not authors or not year:
            return None
        if page["title"] and not _same_title(page["title"], title):
            return None
        if page["year"] and page["year"] != year:
            return None
        if not _authors_agree(page["authors"], authors):
            return None
        containers = record.get("container-title") or [page["container_title"]]
        return Source(
            id=_stable_id(final_url),
            authors=authors,
            title=title,
            year=year,
            url=final_url,
            doi=str(record.get("DOI") or doi),
            container_title=containers[0] or None,
            summary=page["summary"] or str(result.get("body", ""))[:1400],
            verification_basis="crossref",
        )

    if not page["title"] or not page["authors"] or not page["year"]:
        return _source_from_search_result(result, final_url, result_title)
    return Source(
        id=_stable_id(final_url),
        authors=page["authors"],
        title=page["title"],
        year=page["year"],
        url=final_url,
        container_title=page["container_title"] or None,
        summary=page["summary"] or str(result.get("body", ""))[:1400],
        verification_basis="publisher_metadata",
    )


def _source_from_search_result(result: dict[str, Any], result_url: str, result_title: str) -> Source | None:
    if not result_title:
        return None
    try:
        record = _crossref_search(result_title)
    except requests.RequestException:
        return None
    if not record:
        return None
    title = _crossref_title(record)
    authors = _crossref_authors(record)
    year = _crossref_year(record)
    doi = str(record.get("DOI") or "")
    if not title or not authors or not year or not doi:
        return None
    container = record.get("container-title") or []
    return Source(
        id=_stable_id(result_url),
        authors=authors,
        title=title,
        year=year,
        url=result_url,
        doi=doi,
        container_title=container[0] if container else None,
        summary=str(result.get("body", ""))[:1400],
        verification_basis="search_result_crossref",
        search_result_title=result_title,
    )


def search_sources(topic: str, max_results: int = 8) -> list[Source]:
    results = DDGS().text(f"{topic} scientific research study", max_results=max_results, safesearch="moderate")
    sources: list[Source] = []
    seen: set[str] = set()
    for result in results:
        source = _source_from_result(result)
        if source and source.id not in seen:
            sources.append(source)
            seen.add(source.id)
        if len(sources) == 6:
            break
    return sources


def verify_source(source: Source) -> Verification:
    try:
        html, final_url = _fetch_page(str(source.url))
    except requests.RequestException as error:
        if source.verification_basis == "search_result_crossref" and source.doi:
            return _verify_search_result_source(source, str(error))
        return Verification(source.id, False, False, False, False, f"Source could not be reopened: {error}")
    page = _page_metadata(html, final_url)

    if source.doi:
        record = _crossref(source.doi)
        if not record:
            return Verification(source.id, False, False, False, False, "DOI did not resolve in Crossref")
        actual_authors = _crossref_authors(record)
        actual_title = _crossref_title(record)
        actual_year = _crossref_year(record)
        doi_on_page = page["doi"] and _normalize(page["doi"]) == _normalize(source.doi)
        author_ok = (
            {_surname(name) for name in source.authors} == {_surname(name) for name in actual_authors}
            and _authors_agree(page["authors"], actual_authors)
        )
        title_ok = _same_title(source.title, actual_title) and (
            not page["title"] or _same_title(page["title"], actual_title)
        )
        year_ok = source.year == actual_year and (not page["year"] or page["year"] == actual_year)
        locator_ok = bool(doi_on_page)
        basis = "publisher page DOI and matching Crossref record"
    else:
        actual_authors = page["authors"]
        author_ok = _authors_agree(source.authors, actual_authors) and bool(actual_authors)
        title_ok = bool(page["title"] and _same_title(source.title, page["title"]))
        year_ok = source.year == page["year"]
        original = urlparse(str(source.url))._replace(fragment="").geturl()
        current = urlparse(final_url)._replace(fragment="").geturl()
        locator_ok = original == current
        basis = "publisher page citation metadata"

    status = "All bibliographic fields matched" if all((author_ok, title_ok, year_ok, locator_ok)) else "One or more fields did not match"
    return Verification(source.id, author_ok, title_ok, year_ok, locator_ok, f"{status}; checked against {basis}.")


def _verify_search_result_source(source: Source, fetch_error: str) -> Verification:
    record = _crossref(source.doi or "")
    if not record:
        return Verification(source.id, False, False, False, False, "Publisher page was blocked and DOI did not resolve")
    actual_authors = _crossref_authors(record)
    actual_title = _crossref_title(record)
    actual_year = _crossref_year(record)
    try:
        results = DDGS().text(f'"{source.title}" {source.year}', max_results=8, safesearch="moderate")
    except Exception as error:
        return Verification(source.id, False, False, False, False, f"Publisher page was blocked; source re-search failed: {error}")

    original_url = _normalize(str(source.url).rstrip("/"))
    matched_result = any(
        _normalize(str(result.get("href") or result.get("url") or "").rstrip("/")) == original_url
        and _same_title(str(result.get("title") or ""), source.search_result_title or source.title)
        for result in results
    )
    author_ok = {_surname(name) for name in source.authors} == {_surname(name) for name in actual_authors}
    title_ok = _same_title(source.title, actual_title) and _same_title(source.search_result_title or source.title, actual_title)
    year_ok = source.year == actual_year
    locator_ok = bool(record.get("DOI") and _normalize(str(record["DOI"])) == _normalize(source.doi or "") and matched_result)
    detail = "Checked against the exact web-search result and matching Crossref DOI record"
    if not all((author_ok, title_ok, year_ok, locator_ok)):
        detail = f"Search result/Crossref check mismatch; publisher returned: {fetch_error}"
    return Verification(source.id, author_ok, title_ok, year_ok, locator_ok, detail)


def verify_external_reference(
    label: str,
    authors: list[str],
    title: str,
    year: int,
    doi: str | None,
    url: str | None,
) -> Verification:
    query = f'"{title}" {authors[0] if authors else ""} {year}'
    try:
        results = DDGS().text(query, max_results=5, safesearch="moderate")
    except Exception as error:
        return Verification(label, False, False, False, False, f"Search verification failed: {error}")

    candidates = [source for result in results if (source := _source_from_result(result))]
    if not candidates:
        return Verification(label, False, False, False, False, "No matching original source could be verified")
    candidate = max(candidates, key=lambda source: SequenceMatcher(None, _normalize(title), _normalize(source.title)).ratio())
    author_ok = _authors_agree(authors, candidate.authors)
    title_ok = _same_title(title, candidate.title)
    year_ok = year == candidate.year
    if doi:
        locator_ok = bool(candidate.doi and _normalize(doi) == _normalize(candidate.doi))
    else:
        locator_ok = bool(url and _normalize(url.rstrip("/")) == _normalize(str(candidate.url).rstrip("/")))
    status = "Reference matched a search result and source metadata" if all((author_ok, title_ok, year_ok, locator_ok)) else "Reference is incorrect or incomplete"
    return Verification(label, author_ok, title_ok, year_ok, locator_ok, status)