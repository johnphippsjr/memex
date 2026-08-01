import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from memex.synthesizer.commit import extract_decisions


def _mock_completion(content: str):
    """Build a fake OpenAI ChatCompletion-shaped response carrying ``content``
    as ``choices[0].message.content`` (what extract_decisions reads)."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_trivial_commit_filter_skips_api_call():
    trivial_messages = ["wip", "WIP", "fix", "fmt", "lint", "merge", "typo", "bump", "style"]

    with patch("memex.synthesizer.commit.openai.AsyncOpenAI") as mock_openai_client:
        for msg in trivial_messages:
            result = await extract_decisions(msg, "", "abc123")
            assert result == []
            assert not mock_openai_client.called

@pytest.mark.asyncio
async def test_empty_commit_message_returns_empty_list():
    with patch("memex.synthesizer.commit.openai.AsyncOpenAI") as mock_openai_client:
        result = await extract_decisions("", "", "abc123")
        assert result == []
        assert not mock_openai_client.called

@pytest.mark.asyncio
async def test_whitespace_only_message_returns_empty_list():
    with patch("memex.synthesizer.commit.openai.AsyncOpenAI") as mock_openai_client:
        result = await extract_decisions("   \n\t  ", "", "abc123")
        assert result == []
        assert not mock_openai_client.called

@pytest.mark.asyncio
async def test_rate_limit_retries_with_backoff_then_returns_empty():
    with patch("memex.synthesizer.commit.openai.AsyncOpenAI") as mock_openai_client:
        mock_instance = mock_openai_client.return_value
        mock_instance.chat.completions.create = AsyncMock(
            side_effect=Exception("429 Resource Exhausted")
        )

        # We need to speed up the test by mocking asyncio.sleep
        with patch("memex.synthesizer.commit.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await extract_decisions("Real feature commit", "Diff summary", "abc123")

            assert result == []
            # Check for 3 retries
            assert mock_instance.chat.completions.create.call_count == 3
            assert mock_sleep.call_count == 3 # 2s, 3s, 5s (for attempts 0, 1, 2)

@pytest.mark.asyncio
async def test_extract_decisions_success():
    with patch("memex.synthesizer.commit.openai.AsyncOpenAI") as mock_openai_client:
        mock_instance = mock_openai_client.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(
                '{"decisions": [{"text": "D1", "rationale": "R1", "scope": "local"}]}'
            )
        )

        result = await extract_decisions("commit msg", "diff", "sha123")
        assert len(result) == 1
        assert result[0].text == "D1"


@pytest.mark.asyncio
async def test_synthesised_decision_starts_unvalidated_at_0_6():
    """v0.3.0 Phase 8 (Hallucination Mitigation): every watcher-synthesised
    Decision must come back with validated=False, base_confidence=0.6, and
    source='watcher'. These are the anchors the TempValid two-regime decay
    depends on."""
    with patch("memex.synthesizer.commit.openai.AsyncOpenAI") as mock_openai_client:
        mock_instance = mock_openai_client.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(
                '{"decisions": [{"text": "switched to EdDSA", '
                '"rationale": "RSA rotation too operationally complex", '
                '"scope": "module"}]}'
            )
        )

        result = await extract_decisions(
            "refactor auth: switched from RS256 to EdDSA", "diff", "abc12345"
        )

        assert len(result) == 1
        d = result[0]
        assert d.validated is False, "watcher-synthesised decisions must start unvalidated"
        assert d.base_confidence == 0.6
        assert d.source == "watcher"
        assert d.source_commit == "abc12345"
