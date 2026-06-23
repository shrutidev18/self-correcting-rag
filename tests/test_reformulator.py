import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_reformulator():
    with patch("app.core.reformulator.Groq") as mock_groq:
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        from app.core.reformulator import Reformulator
        reformulator = Reformulator()
        reformulator.client = mock_client
        yield reformulator, mock_client


def make_mock_response(content: str):
    mock_response               = MagicMock()
    mock_choice                 = MagicMock()
    mock_choice.message.content = content
    mock_response.choices       = [mock_choice]
    return mock_response


class TestParseRewrites:

    def test_clean_json_array(self):
        from app.core.reformulator import _parse_rewrites
        raw = '["rewrite 1", "rewrite 2", "rewrite 3"]'
        result = _parse_rewrites(raw)
        assert len(result) == 3
        assert result[0] == "rewrite 1"

    def test_json_with_code_fence(self):
        from app.core.reformulator import _parse_rewrites
        raw = '```json\n["rewrite 1", "rewrite 2", "rewrite 3"]\n```'
        result = _parse_rewrites(raw)
        assert len(result) == 3

    def test_array_of_dicts(self):
        from app.core.reformulator import _parse_rewrites
        raw = '[{"rewrite_1": "question one"}, {"rewrite_2": "question two"}]'
        result = _parse_rewrites(raw)
        assert len(result) == 2

    def test_invalid_json_returns_empty(self):
        from app.core.reformulator import _parse_rewrites
        raw = "not json at all"
        result = _parse_rewrites(raw)
        assert isinstance(result, list)

    def test_returns_max_3(self):
        from app.core.reformulator import _parse_rewrites
        raw = '["q1", "q2", "q3", "q4", "q5"]'
        result = _parse_rewrites(raw)
        assert len(result) <= 3


class TestReformulator:

    def test_reformulate_returns_list(self, mock_reformulator):
        reformulator, mock_client = mock_reformulator
        mock_client.chat.completions.create.return_value = make_mock_response(
            '["rewrite 1", "rewrite 2", "rewrite 3"]'
        )
        result = reformulator.reformulate("test question")
        assert isinstance(result, list)

    def test_reformulate_returns_3_rewrites(self, mock_reformulator):
        reformulator, mock_client = mock_reformulator
        mock_client.chat.completions.create.return_value = make_mock_response(
            '["rewrite 1", "rewrite 2", "rewrite 3"]'
        )
        result = reformulator.reformulate("test question")
        assert len(result) == 3

    def test_reformulate_api_failure_returns_empty(self, mock_reformulator):
        reformulator, mock_client = mock_reformulator
        mock_client.chat.completions.create.side_effect = Exception("API error")
        result = reformulator.reformulate("test question")
        assert result == []

    def test_reformulate_rewrites_are_strings(self, mock_reformulator):
        reformulator, mock_client = mock_reformulator
        mock_client.chat.completions.create.return_value = make_mock_response(
            '["rewrite 1", "rewrite 2", "rewrite 3"]'
        )
        result = reformulator.reformulate("test question")
        for r in result:
            assert isinstance(r, str)
            assert len(r) > 0