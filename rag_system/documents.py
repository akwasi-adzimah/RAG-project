"""Load PDF/DOCX files and split them into citation-ready chunks.

Chunks never cross page boundaries, so every chunk carries an exact page
number. Splitting happens on sentence boundaries so a retrieved chunk reads as
complete prose rather than starting or ending mid-sentence.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pypdfium2 as pdfium
from docx import Document

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx")
PAGE_CONTEXT_WORDS = 12

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“‘(\[•-])")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    id: int
    source: str
    page: int | None
    text: str
    context: str = ""  # opening words of the page (usually its heading), used for indexing only

    @property
    def label(self) -> str:
        return f"{self.source}, p. {self.page}" if self.page else self.source

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DocumentInfo:
    name: str
    pages: int
    words: int
    chunks: int


@dataclass
class Corpus:
    chunks: list[Chunk]
    documents: list[DocumentInfo]
    skipped: list[str]  # files with no extractable text (e.g. scanned PDFs)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def clean_page_text(raw: str) -> str:
    """Normalise extracted text and drop duplicated lines.

    Some PDFs (e.g. slide exports) contain the same text twice on a page - once
    as visible text and once in an accessibility layer. Repeated long lines are
    removed so they don't dominate retrieval scores.
    """
    text = _HYPHEN_BREAK.sub(r"\1\2", raw.replace("\r", "\n"))
    seen: set[str] = set()
    kept: list[str] = []
    for line in text.split("\n"):
        line = _WHITESPACE.sub(" ", line).strip()
        if not line:
            continue
        key = line.lower()
        if len(line) > 30 and key in seen:
            continue
        seen.add(key)
        kept.append(line)
    return " ".join(kept)


def extract_pdf(path: Path) -> list[tuple[int | None, str]]:
    pdf = pdfium.PdfDocument(str(path))
    try:
        pages = []
        for i in range(len(pdf)):
            textpage = pdf[i].get_textpage()
            text = clean_page_text(textpage.get_text_range())
            if text:
                pages.append((i + 1, text))
        return pages
    finally:
        pdf.close()


def extract_docx(path: Path) -> list[tuple[int | None, str]]:
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = clean_page_text("\n".join(parts))
    return [(None, text)] if text else []


def extract_file(path: Path) -> list[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".docx":
        return extract_docx(path)
    return []


def source_files(docs_dir: Path) -> list[Path]:
    return sorted(
        p for p in docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith("~$")
    )


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def split_sentences(text: str, max_words: int) -> list[str]:
    """Split on sentence boundaries; hard-wrap any 'sentence' longer than max_words
    (PDF tables and bullet lists often have no terminal punctuation)."""
    out: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        words = sentence.split()
        for i in range(0, len(words), max_words):
            piece = " ".join(words[i:i + max_words])
            if piece:
                out.append(piece)
    return out


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    sentences = split_sentences(text, chunk_words)
    chunks: list[str] = []
    current: list[str] = []
    count = 0

    for sentence in sentences:
        n = len(sentence.split())
        if current and count + n > chunk_words:
            chunks.append(" ".join(current))
            # Carry trailing sentences forward as overlap.
            overlap: list[str] = []
            kept = 0
            for prev in reversed(current):
                w = len(prev.split())
                if kept + w > overlap_words:
                    break
                overlap.insert(0, prev)
                kept += w
            current, count = overlap, kept
        current.append(sentence)
        count += n

    if current:
        tail = " ".join(current)
        # Don't emit a final chunk that is nothing but overlap of the previous one.
        if not chunks or not chunks[-1].endswith(tail):
            chunks.append(tail)
    return chunks


def load_corpus(docs_dir: Path, chunk_words: int, overlap_words: int) -> Corpus:
    chunks: list[Chunk] = []
    documents: list[DocumentInfo] = []
    skipped: list[str] = []

    for path in source_files(docs_dir):
        try:
            pages = extract_file(path)
        except Exception as exc:  # noqa: BLE001 - corrupt / encrypted files shouldn't sink the whole index
            log.warning("Could not read %s: %s", path.name, exc)
            skipped.append(path.name)
            continue
        if not pages:
            log.warning("No extractable text in %s (scanned PDF? needs OCR)", path.name)
            skipped.append(path.name)
            continue

        start = len(chunks)
        words = 0
        for page, text in pages:
            words += len(text.split())
            context = " ".join(text.split()[:PAGE_CONTEXT_WORDS])
            for piece in chunk_text(text, chunk_words, overlap_words):
                chunks.append(Chunk(id=len(chunks), source=path.name, page=page,
                                    text=piece, context=context))
        documents.append(DocumentInfo(
            name=path.name,
            pages=max((p for p, _ in pages if p), default=1),
            words=words,
            chunks=len(chunks) - start,
        ))

    return Corpus(chunks=chunks, documents=documents, skipped=skipped)
