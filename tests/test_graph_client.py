import os
import pytest
from memex.graph.client import get_graph_client

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("FALKOR_HOST_LIVE"),
    reason="FALKOR_HOST_LIVE must be set to a live FalkorDB host to run integration tests"
)
async def test_graph_client_initialization():
    """
    Test that the Graphiti client initializes and can connect to FalkorDB.
    """
    try:
        client = await get_graph_client()
        assert client is not None
        # Basic check to see if we can talk to the driver
        assert client.driver is not None
        print("Graphiti client initialized and connected successfully.")
    except Exception as e:
        pytest.fail(f"Graphiti client failed to initialize: {e}")


# ---------------------------------------------------------------------------
# Board #788 — the temperature pin must actually REACH the LLM client.
# A config field nobody passes through is not a feature; this is the caller
# check, not a restatement of the default.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_and_reranker_configs_receive_pinned_temperature(monkeypatch):
    """get_instance() must pass config.llm_temperature into BOTH LLMConfig
    constructions (extraction client and reranker). Without this, graphiti-core
    falls back to its own DEFAULT_TEMPERATURE of 1 and every extraction samples
    -- the exact condition #788 measured at a 20% malformed-output rate."""
    import memex.graph.client as gc

    captured = []

    class _FakeLLMConfig:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    class _Stub:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(gc, "LLMConfig", _FakeLLMConfig)
    monkeypatch.setattr(gc, "CompatFalkorDriver", _Stub)
    monkeypatch.setattr(gc, "OpenAIGenericClient", _Stub)
    monkeypatch.setattr(gc, "SalvagingLocalClient", _Stub)  # #790: the LLM client is this now
    monkeypatch.setattr(gc, "OpenAIEmbedder", _Stub)
    monkeypatch.setattr(gc, "OpenAIEmbedderConfig", _Stub)
    monkeypatch.setattr(gc, "OpenAIRerankerClient", _Stub)
    monkeypatch.setattr(gc, "Graphiti", _Stub)
    monkeypatch.setattr(gc, "apply_all_patches", lambda *a, **k: None)

    class _Cfg:
        falkor_host = "h"
        falkor_port = 6379
        falkor_graph = "g"
        litellm_api_key = "k"
        litellm_base_url = "http://x"
        litellm_model = "m"
        llm_structured_output_mode = "json_object"
        llm_temperature = 0.0
        embedding_model = "bge-m3"
        embedding_dim = 1024

    monkeypatch.setattr(gc, "get_config", lambda: _Cfg())
    monkeypatch.setattr(gc.GraphClient, "_instance", None)

    await gc.GraphClient.get_instance()
    gc.GraphClient._instance = None

    assert len(captured) == 2, f"expected 2 LLMConfig builds, got {len(captured)}"
    for kwargs in captured:
        assert "temperature" in kwargs, "LLMConfig built without temperature"
        assert kwargs["temperature"] == 0.0
