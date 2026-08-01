import os
import pytest
from pathlib import Path

# Force MEMEX_REGISTRY_PATH to a non-existent temp file during import
# to prevent tests from interacting with the user's live registry.
os.environ["MEMEX_REGISTRY_PATH"] = str(Path(__file__).parent / "test_registry_tmp.json")

@pytest.fixture(scope="session", autouse=True)
def clean_registry_path(tmp_path_factory):
    temp_file = tmp_path_factory.mktemp("registry") / "registry.json"
    from memex.watcher import registry
    registry.REGISTRY_PATH = temp_file
    yield


@pytest.fixture(autouse=True)
def _default_falkor_litellm_env(monkeypatch):
    """Safe dummy FalkorDB/LiteLLM config so a bare ``get_config()`` call
    succeeds by default across the suite (falkordb-litellm fork — no live
    backend is required for unit tests). Mirrors the old NEO4J_*/GEMINI_*
    env vars this fork removes; individual tests that need specific values
    or full isolation still patch ``get_config``/construct ``Config(...)``
    directly, or ``monkeypatch.delenv`` these to exercise the
    missing-config path.
    """
    monkeypatch.setenv("FALKOR_HOST", "localhost")
    monkeypatch.setenv("FALKOR_PORT", "6379")
    monkeypatch.setenv("FALKOR_GRAPH", "test-graph")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_MODEL", "test-model")
    monkeypatch.setenv("UNIFIED_GROUP_ID", "test-group")

    from memex import config as _config_mod
    monkeypatch.setattr(_config_mod, "_config", None)
    yield
    monkeypatch.setattr(_config_mod, "_config", None)
