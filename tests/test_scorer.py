import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_scorer():
    with patch("app.core.scorer.Groq") as mock_groq:
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        from app.core.scorer import Scorer
        scorer = Scorer(threshold=3.0)
        scorer.client = mock_client
        yield scorer, mock_client


def make_mock_response(content: str, tokens: int = 100):
    """Helper to build a fake Groq API response."""
    mock_response          = MagicMock()
    mock_choice            = MagicMock()
    mock_choice.message.content = content
    mock_response.choices  = [mock_choice]
    mock_response.usage.total_tokens = tokens
    return mock_response


class TestParseScore:

    def test_clean_digit(self):
        from app.core.scorer import _parse_score
        assert _parse_score("4") == 4

    def test_score_with_label(self):
        from app.core.scorer import _parse_score
        assert _parse_score("Score: 4") == 4

    def test_score_with_fraction(self):
        from app.core.scorer import _parse_score
        assert _parse_score("4/5") == 4

    def test_score_in_sentence(self):
        from app.core.scorer import _parse_score
        assert _parse_score("I'd give it a 3") == 3

    def test_invalid_falls_back_to_3(self):
        from app.core.scorer import _parse_score
        assert _parse_score("unknown") == 3


class TestScorer:

    def test_score_returns_required_fields(self, mock_scorer):
        scorer, mock_client = mock_scorer
        mock_client.chat.completions.create.return_value = make_mock_response("4")

        chunks = [{"id": f"chunk_{i}", "text": f"text {i}"} for i in range(3)]
        result = scorer.score("test question", chunks)

        assert "scores"      in result
        assert "mean"        in result
        assert "quality"     in result
        assert "threshold"   in result
        assert "latency_ms"  in result
        assert "token_count" in result

    def test_score_good_quality(self, mock_scorer):
        scorer, mock_client = mock_scorer
        mock_client.chat.completions.create.return_value = make_mock_response("4")

        chunks = [{"id": f"c{i}", "text": f"text {i}"} for i in range(5)]
        result = scorer.score("question", chunks)

        assert result["quality"] == "good"
        assert result["mean"] >= 3.0

    def test_score_poor_quality(self, mock_scorer):
        scorer, mock_client = mock_scorer
        mock_client.chat.completions.create.return_value = make_mock_response("2")

        chunks = [{"id": f"c{i}", "text": f"text {i}"} for i in range(5)]
        result = scorer.score("question", chunks)

        assert result["quality"] == "poor"
        assert result["mean"] < 3.0

    def test_empty_chunks_returns_poor(self, mock_scorer):
        scorer, _ = mock_scorer
        result = scorer.score("question", [])
        assert result["quality"] == "poor"
        assert result["scores"] == []

    def test_score_count_matches_chunks(self, mock_scorer):
        scorer, mock_client = mock_scorer
        mock_client.chat.completions.create.return_value = make_mock_response("3")

        chunks = [{"id": f"c{i}", "text": f"text {i}"} for i in range(4)]
        result = scorer.score("question", chunks)

        assert len(result["scores"]) == 4

    def test_llm_failure_defaults_to_3(self, mock_scorer):
        scorer, mock_client = mock_scorer
        mock_client.chat.completions.create.side_effect = Exception("API error")

        chunks = [{"id": "c1", "text": "some text"}]
        result = scorer.score("question", chunks)

        assert result["scores"] == [3]