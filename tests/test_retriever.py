#Tests prove your code works correctly. This file tests every function in retriever.py — chunking, indexing, and retrieval — without touching your real database or downloading any models.

import pytest
import chromadb
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_retriever():
    from app.core.retriever import Retriever

    with patch("app.core.retriever.SentenceTransformer") as mock_st, \
         patch("app.core.retriever.chromadb.PersistentClient") as mock_chroma:

        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = lambda texts, **kw: np.ones((len(texts) if isinstance(texts, list) else 1, 384))
        mock_st.return_value = mock_embedder

        mock_chroma.return_value = chromadb.EphemeralClient()

        retriever = Retriever()
        retriever.embedder = mock_embedder
        yield retriever


class TestChunking:

    def test_basic_chunking(self, mock_retriever):
        long_text = " ".join(["word"] * 600)
        chunks = mock_retriever._chunk_text(long_text)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self, mock_retriever):
        short_text = "This is a very short text."
        chunks = mock_retriever._chunk_text(short_text)
        assert len(chunks) == 1

    def test_chunks_have_overlap(self, mock_retriever):
        text = " ".join([f"word{i}" for i in range(400)])
        chunks = mock_retriever._chunk_text(text)
        if len(chunks) >= 2:
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            assert len(words_0 & words_1) > 0

    def test_empty_text(self, mock_retriever):
        chunks = mock_retriever._chunk_text("")
        assert chunks == []


class TestIndexing:

    def test_index_single_document(self, mock_retriever):
        before = mock_retriever.count()
        mock_retriever.index_documents([
            {"id": "test_001", "text": "Python is a programming language.", "title": "Python"}
        ])
        assert mock_retriever.count() > before

    def test_index_empty_list(self, mock_retriever):
        count_before = mock_retriever.count()
        mock_retriever.index_documents([])
        assert mock_retriever.count() == count_before

    def test_no_duplicates_on_reindex(self, mock_retriever):
        doc = {"id": "dedup_test", "text": "Unique content for dedup test.", "title": ""}
        mock_retriever.index_documents([doc])
        count_after_first = mock_retriever.count()
        mock_retriever.index_documents([doc])
        assert mock_retriever.count() == count_after_first


class TestRetrieval:

    def test_retrieve_returns_k_results(self, mock_retriever):
        docs = [
            {"id": f"doc_{i}", "text": f"Document number {i} about various topics.", "title": ""}
            for i in range(10)
        ]
        mock_retriever.index_documents(docs)
        results = mock_retriever.retrieve("test query", k=3)
        assert len(results) == 3

    def test_result_has_required_fields(self, mock_retriever):
        docs = [{"id": "struct_test", "text": "Testing result structure carefully.", "title": "Test"}]
        mock_retriever.index_documents(docs)
        results = mock_retriever.retrieve("structure test", k=1)
        r = results[0]
        assert "id"     in r
        assert "text"   in r
        assert "score"  in r
        assert "doc_id" in r
        assert "title"  in r

    def test_scores_between_0_and_1(self, mock_retriever):
        docs = [{"id": f"score_{i}", "text": f"Score test document {i}.", "title": ""} for i in range(5)]
        mock_retriever.index_documents(docs)
        results = mock_retriever.retrieve("score test", k=3)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_empty_db_raises_error(self, mock_retriever):
        from app.core.retriever import Retriever
        with patch("app.core.retriever.SentenceTransformer"), \
             patch("app.core.retriever.chromadb.PersistentClient") as mock_chroma:

            mock_client     = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_client

            empty_retriever = Retriever()
            empty_retriever.embedder = MagicMock()
            empty_retriever.embedder.encode.side_effect = lambda texts, **kw: np.ones((len(texts) if isinstance(texts, list) else 1, 384))

            with pytest.raises(RuntimeError, match="No documents indexed"):
                empty_retriever.retrieve("anything")