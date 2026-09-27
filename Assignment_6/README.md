# Grounded Article Studio

A Streamlit application that searches the web for scientific or technical sources, creates a Pydantic-validated article, checks its references against publisher pages and Crossref, and exports the article as a PDF. It can also generate an unsearched comparison draft and check those references afterward.

## Run

Python 3.11 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-key"
streamlit run app.py
```

You can also enter the API key in the sidebar. The key is sent only to OpenAI and is not saved by the app. Web search uses DuckDuckGo results; bibliographic verification uses publisher-page metadata and Crossref.

## Validation

The grounded generation prompt can select only IDs returned from verified search results. Pydantic rejects unknown citations, unused references, malformed article sections, and duplicate source IDs before PDF rendering. The PDF is withheld if a second source-page verification fails.

The no-search draft is generated without search results or tools. After generation, its references are searched individually and compared by author, title, year, and DOI or URL; the comparison tab reports exact genuine and incorrect/unverified counts.

Run model validation tests with:

```powershell
python -m unittest discover -s tests
```