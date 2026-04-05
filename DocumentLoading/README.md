# Document Loading

Ingests raw content from various sources into a unified `Document` format before chunking and embedding.

## Supported Sources

| Source | Loader |
|---|---|
| PDF | `PyPDFLoader`, `PDFMinerLoader` |
| Web / HTML | `WebBaseLoader`, `RecursiveUrlLoader` |
| Plain text / Markdown | `TextLoader` |
| Word (.docx) | `Docx2txtLoader` |
| CSV / Excel | `CSVLoader`, `UnstructuredExcelLoader` |
| Notion / Confluence | API-based loaders |
| YouTube transcripts | `YoutubeLoader` |

## Pipeline Position

```
[Source] → [Document Loading] → [Chunking] → [Embedding] → [VectorStore]
```

## Output Schema

Every loader produces a list of `Document` objects:

```python
Document(
    page_content: str,
    metadata: {
        "source": str,       # file path or URL
        "page": int,         # page number if applicable
        "doc_id": str,       # stable unique ID for dedup
        ...
    }
)
```

## Key Concerns

- **Deduplication** — use `doc_id` in metadata to skip already-ingested documents
- **Encoding** — normalize to UTF-8 before passing downstream
- **Large files** — stream or paginate; avoid loading entire files into memory
