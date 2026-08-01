"""Unit tests for ``memex.graph.cluster_summary``.

Tests for synthesize_cluster_summary() and refresh_cluster_summaries()
without requiring a live FalkorDB, LiteLLM gateway or network access.
"""

from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from memex.graph.cluster_summary import (
    synthesize_cluster_summary,
    refresh_cluster_summaries,
)


def _mock_completion(text: str):
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.asyncio
async def test_synthesize_returns_single_sentence():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion("This is a single sentence summary of the cluster.")
    )

    with patch("memex.graph.cluster_summary.openai.AsyncOpenAI", return_value=mock_client), \
         patch("memex.graph.cluster_summary.get_config") as mock_get_config:
        mock_get_config.return_value = SimpleNamespace(
            litellm_base_url="http://localhost:4000",
            litellm_api_key="fake_key",
            litellm_model="test-model",
        )

        summary = await synthesize_cluster_summary(
            cluster_name="test-cluster",
            member_modules=["a.py", "b.py"],
            decision_texts=["Adopt pattern X", "Use library Y"]
        )

        assert summary == "This is a single sentence summary of the cluster."


@pytest.mark.asyncio
async def test_synthesize_truncates_decisions():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion("Summary sentence.")
    )

    with patch("memex.graph.cluster_summary.openai.AsyncOpenAI", return_value=mock_client), \
         patch("memex.graph.cluster_summary.get_config") as mock_get_config:
        mock_get_config.return_value = SimpleNamespace(
            litellm_base_url="http://localhost:4000",
            litellm_api_key="fake_key",
            litellm_model="test-model",
        )

        # Pass 25 decisions, limit to max_decisions=5
        decisions = [f"Decision {i}" for i in range(25)]
        await synthesize_cluster_summary(
            cluster_name="test-cluster",
            member_modules=["a.py"],
            decision_texts=decisions,
            max_decisions=5
        )

        # Inspect prompt passed to chat.completions.create
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"] if call_args else []
        prompt = messages[0]["content"] if messages else ""
        assert "Decision 0" in prompt
        assert "Decision 4" in prompt
        assert "Decision 5" not in prompt


@pytest.mark.asyncio
async def test_refresh_skips_existing_summaries():
    mock_client = AsyncMock()
    mock_driver = AsyncMock()
    mock_client.driver = mock_driver

    # Mock records returned by execute_query
    mock_record = MagicMock()
    mock_record.data.return_value = {
        "cluster_name": "cluster-1",
        "members": ["module_a.py"],
        "decisions": ["Adopt library X"]
    }
    mock_result = MagicMock()
    mock_result.records = [mock_record]
    mock_driver.execute_query.return_value = mock_result

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion("Summary 1")
    )

    with patch("memex.graph.client.get_graph_client", return_value=mock_client), \
         patch("memex.graph.cluster_summary.openai.AsyncOpenAI", return_value=mock_openai_client), \
         patch("memex.graph.cluster_summary.get_config") as mock_get_config:
        mock_get_config.return_value = SimpleNamespace(
            litellm_base_url="http://localhost:4000",
            litellm_api_key="fake_key",
            litellm_model="test-model",
            repo_root="/fake",
        )

        res = await refresh_cluster_summaries("/fake", force=False)
        assert res == {"cluster-1": "Summary 1"}

        # Check that the first query includes "AND c.summary IS NULL"
        args = mock_driver.execute_query.call_args_list[0]
        query = args[0][0]
        assert "AND c.summary IS NULL" in query


@pytest.mark.asyncio
async def test_refresh_force_regenerates_all():
    mock_client = AsyncMock()
    mock_driver = AsyncMock()
    mock_client.driver = mock_driver

    # Mock records returned by execute_query
    mock_record = MagicMock()
    mock_record.data.return_value = {
        "cluster_name": "cluster-1",
        "members": ["module_a.py"],
        "decisions": ["Adopt library X"]
    }
    mock_result = MagicMock()
    mock_result.records = [mock_record]
    mock_driver.execute_query.return_value = mock_result

    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_completion("Summary 1")
    )

    with patch("memex.graph.client.get_graph_client", return_value=mock_client), \
         patch("memex.graph.cluster_summary.openai.AsyncOpenAI", return_value=mock_openai_client), \
         patch("memex.graph.cluster_summary.get_config") as mock_get_config:
        mock_get_config.return_value = SimpleNamespace(
            litellm_base_url="http://localhost:4000",
            litellm_api_key="fake_key",
            litellm_model="test-model",
            repo_root="/fake",
        )

        res = await refresh_cluster_summaries("/fake", force=True)
        assert res == {"cluster-1": "Summary 1"}

        # Check that the first query does NOT include "AND c.summary IS NULL"
        args = mock_driver.execute_query.call_args_list[0]
        query = args[0][0]
        assert "AND c.summary IS NULL" not in query


@pytest.mark.asyncio
async def test_refresh_skips_empty_clusters():
    mock_client = AsyncMock()
    mock_driver = AsyncMock()
    mock_client.driver = mock_driver

    # Cluster with no member modules
    mock_record = MagicMock()
    mock_record.data.return_value = {
        "cluster_name": "cluster-1",
        "members": [],
        "decisions": []
    }
    mock_result = MagicMock()
    mock_result.records = [mock_record]
    mock_driver.execute_query.return_value = mock_result

    with patch("memex.graph.client.get_graph_client", return_value=mock_client):
        res = await refresh_cluster_summaries("/fake", force=True)
        assert res == {}
        # verify no write-back query (execute_query only called once for fetch)
        assert mock_driver.execute_query.call_count == 1
