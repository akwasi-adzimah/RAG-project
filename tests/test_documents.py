"""Extraction clean-up and chunking."""
from itertools import pairwise

from docx import Document

from rag_system.documents import (
    chunk_text,
    clean_page_text,
    load_corpus,
    split_sentences,
)


def test_clean_page_text_drops_duplicated_long_lines():
    line = "This sentence is long enough to count as a duplicate line."
    raw = f"Heading\n{line}\n\n{line}\nShort\nShort"
    assert clean_page_text(raw) == f"Heading {line} Short Short"


def test_clean_page_text_rejoins_hyphenated_words():
    assert clean_page_text("develop-\nment team") == "development team"


def test_split_sentences_hard_wraps_long_runs():
    text = " ".join(f"w{i}" for i in range(25))  # no punctuation, like a PDF table
    parts = split_sentences(text, max_words=10)
    assert [len(p.split()) for p in parts] == [10, 10, 5]


def test_chunks_respect_size_and_overlap():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(40)]
    chunks = chunk_text(" ".join(sentences), chunk_words=50, overlap_words=14)

    assert len(chunks) > 1
    assert all(len(c.split()) <= 50 for c in chunks)
    # Each chunk after the first starts with the last sentences of the previous one.
    for prev, nxt in pairwise(chunks):
        assert prev.split(". ")[-1].rstrip(".") in nxt
    # Nothing is lost.
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_chunks_end_on_sentence_boundaries():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
    for chunk in chunk_text(text, chunk_words=6, overlap_words=0):
        assert chunk.endswith(".")


def test_short_text_is_single_chunk():
    assert chunk_text("Just one sentence.", 200, 40) == ["Just one sentence."]


def test_load_corpus_reads_docx_and_skips_unsupported(tmp_path):
    doc = Document()
    doc.add_paragraph("The hardship statement explains the circumstances.")
    doc.save(tmp_path / "statement.docx")
    (tmp_path / "notes.txt").write_text("ignored")

    corpus = load_corpus(tmp_path, chunk_words=200, overlap_words=40)

    assert [d.name for d in corpus.documents] == ["statement.docx"]
    assert corpus.chunks[0].page is None
    assert "hardship statement" in corpus.chunks[0].text
    assert corpus.chunks[0].label == "statement.docx"


def test_corrupt_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"not really a pdf")
    corpus = load_corpus(tmp_path, chunk_words=200, overlap_words=40)
    assert corpus.chunks == []
    assert corpus.skipped == ["broken.pdf"]
