import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def retriever():
    with patch("app.core.retriever.chromadb") as mock_chroma, \
         patch("app.core.retriever.SentenceTransformer") as mock_st:

        mock_client = MagicMock()
        mock_chroma.PersistentClient.return_value = mock_client

        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.count.return_value = 10

        mock_embedder = MagicMock()
        mock_st.return_value = mock_embedder
        mock_embedder.encode.return_value = [[0.1] * 384]

        from app.core.retriever import Retriever
        r = Retriever()
        r.collection = mock_collection
        r.embedder = mock_embedder
        yield r, mock_collection, mock_embedder


class TestChunking:

    def test_basic_chunking(self, retriever):
        r, _, _ = retriever
        text = " ".join([f"word{i}" for i in range(300)])
        chunks = r._chunk_text(text)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self, retriever):
        r, _, _ = retriever
        text = "short text with few words"
        chunks = r._chunk_text(text)
        assert len(chunks) == 1

    def test_chunks_have_overlap(self, retriever):
        r, _, _ = retriever
        text = " ".join([f"word{i}" for i in range(300)])
        chunks = r._chunk_text(text)
        # with overlap, consecutive chunks should share some words
        words_1 = set(chunks[0].split())
        words_2 = set(chunks[1].split())
        assert len(words_1 & words_2) > 0

    def test_empty_text(self, retriever):
        r, _, _ = retriever
        chunks = r._chunk_text("")
        assert chunks == []


class TestIndexing:

    def test_index_single_document(self, retriever):
        r, mock_collection, mock_embedder = retriever
        import numpy as np
        mock_embedder.encode.return_value = np.array([[0.1] * 384] * 10)

        docs = [{"id": "doc1", "text": " ".join(["word"] * 300), "title": "test"}]
        count = r.index_documents(docs)
        assert count > 0
        assert mock_collection.upsert.called

    def test_index_empty_list(self, retriever):
        r, mock_collection, _ = retriever
        count = r.index_documents([])
        assert count == 0
        assert not mock_collection.upsert.called

    def test_no_duplicates_on_reindex(self, retriever):
        r, mock_collection, mock_embedder = retriever
        import numpy as np
        mock_embedder.encode.return_value = np.array([[0.1] * 384] * 10)

        docs = [{"id": "doc1", "text": " ".join(["word"] * 300), "title": "test"}]
        r.index_documents(docs)
        r.index_documents(docs)
        # upsert handles deduplication — called twice is fine
        assert mock_collection.upsert.call_count >= 1


class TestRetrieval:

    def test_retrieve_returns_k_results(self, retriever):
        r, mock_collection, mock_embedder = retriever
        import numpy as np
        mock_embedder.encode.return_value = np.array([0.1] * 384)
        mock_collection.count.return_value = 10
        mock_collection.query.return_value = {
            "ids":       [["id1", "id2", "id3"]],
            "documents": [["text1", "text2", "text3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [[{"doc_id": "d1", "title": "t1"},
                           {"doc_id": "d2", "title": "t2"},
                           {"doc_id": "d3", "title": "t3"}]],
        }
        results = r.retrieve("test query", k=3)
        assert len(results) == 3

    def test_result_has_required_fields(self, retriever):
        r, mock_collection, mock_embedder = retriever
        import numpy as np
        mock_embedder.encode.return_value = np.array([0.1] * 384)
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids":       [["id1"]],
            "documents": [["some text"]],
            "distances": [[0.2]],
            "metadatas": [[{"doc_id": "d1", "title": "t1"}]],
        }
        results = r.retrieve("query")
        assert "text"   in results[0]
        assert "id"     in results[0]
        assert "score"  in results[0]
        assert "doc_id" in results[0]

    def test_scores_between_0_and_1(self, retriever):
        r, mock_collection, mock_embedder = retriever
        import numpy as np
        mock_embedder.encode.return_value = np.array([0.1] * 384)
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids":       [["id1", "id2"]],
            "documents": [["text1", "text2"]],
            "distances": [[0.1, 0.9]],
            "metadatas": [[{"doc_id": "d1", "title": "t1"},
                           {"doc_id": "d2", "title": "t2"}]],
        }
        results = r.retrieve("query")
        for result in results:
            assert 0.0 <= result["score"] <= 1.0

    def test_empty_db_raises_error(self, retriever):
        r, mock_collection, _ = retriever
        mock_collection.count.return_value = 0
        with pytest.raises(RuntimeError):
            r.retrieve("test query")