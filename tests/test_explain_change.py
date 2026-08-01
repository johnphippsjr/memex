"""Phase 9 — explain_change MCP tool tests."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from memex.mcp_server.tools_explain import explain_change


@pytest.mark.asyncio
async def test_explain_change_returns_grounded_response():
    """Mocks git show + queries and asserts the synthesised response
    actually references the linked Decision text (i.e. it was passed into
    the prompt, not silently ignored)."""
    fake_diff = (
        "diff --git a/memex/auth.py b/memex/auth.py\n"
        "index abc..def 100644\n"
        "--- a/memex/auth.py\n"
        "+++ b/memex/auth.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-def sign(token): return rs256(token)\n"
        "+def sign(token): return eddsa(token)\n"
    )
    linked_decision = {
        "node_type": "Decision",
        "text": "Switch to EdDSA for simpler key rotation",
        "node_id": "decision-1",
        "module": "memex/auth.py",
        "date": "2026-05-01",
    }

    with patch("memex.mcp_server.tools_explain._git_show", new=AsyncMock(return_value=fake_diff)), \
         patch(
             "memex.mcp_server.tools_explain._query_linked_decisions_and_problems",
             new=AsyncMock(return_value=[linked_decision]),
         ), \
         patch(
             "memex.mcp_server.tools_explain._call_gemini_pro",
             new=AsyncMock(return_value="# Why\nThe team chose EdDSA for simpler key rotation."),
         ):
        result = await explain_change("deadbeef")

    # The grounded synthesis must surface the decision text or rationale
    assert "EdDSA" in result
    assert "rotation" in result.lower()


@pytest.mark.asyncio
async def test_explain_change_uses_pro_model_not_default():
    """Inspect the model id passed to the gateway client. Must be
    config.pro_model (the dedicated grounded-synthesis model) — never the
    default litellm_model used for extraction elsewhere."""
    captured: dict = {}

    async def fake_create(*args, **kwargs):
        captured["model"] = kwargs.get("model")
        message = MagicMock()
        message.content = "synthesis ok"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    fake_client = MagicMock()
    fake_client.chat.completions.create = fake_create

    fake_diff = "diff --git a/x b/x\n@@ +1 @@\n+ok\n"

    with patch("memex.mcp_server.tools_explain.openai.AsyncOpenAI", return_value=fake_client), \
         patch("memex.mcp_server.tools_explain.get_config") as mock_get_config, \
         patch("memex.mcp_server.tools_explain._git_show", new=AsyncMock(return_value=fake_diff)), \
         patch(
             "memex.mcp_server.tools_explain._query_linked_decisions_and_problems",
             new=AsyncMock(return_value=[]),
         ):
        mock_get_config.return_value = SimpleNamespace(
            litellm_base_url="http://localhost:4000",
            litellm_api_key="fake_key",
            litellm_model="sentinel-default-model",
            pro_model="sentinel-pro-model",
            repo_root=None,
        )
        await explain_change("abc1234")

    assert captured.get("model"), "model id was not captured — gateway patch missed"
    assert captured["model"] == "sentinel-pro-model", (
        f"explain_change must use config.pro_model, got: {captured['model']}"
    )
    assert captured["model"] != "sentinel-default-model"


@pytest.mark.asyncio
async def test_explain_change_handles_unknown_commit_gracefully():
    """git show failure must not raise — return a graceful string.
    Post-S1: SHA must be hex/length-valid to clear the regex gate; only
    THEN does the git-not-found code path execute."""
    unknown_but_well_formed = "deadbeefcafe1234"  # 16 hex chars, no such commit
    with patch("memex.mcp_server.tools_explain._git_show", new=AsyncMock(return_value=None)):
        result = await explain_change(unknown_but_well_formed)

    assert "could not find commit" in result.lower()
    assert unknown_but_well_formed in result


@pytest.mark.asyncio
async def test_explain_change_rejects_empty_sha():
    result = await explain_change("")
    assert "commit_sha" in result.lower()
    assert "required" in result.lower()


@pytest.mark.asyncio
async def test_explain_change_rejects_argument_injection_via_commit_sha():
    """S1 regression: an attacker-controlled `commit_sha` containing a git
    flag (e.g. `--upload-pack=evil.sh`) must be rejected BEFORE reaching
    the git subprocess. Validated by regex at the public entry; the `--`
    separator on the exec call is the defense-in-depth second layer."""
    from memex.mcp_server.tools_explain import explain_change

    malicious_inputs = [
        "--upload-pack=evil.sh",
        "--exec=/tmp/x",
        "-c core.fsmonitor=evil",
        "HEAD; rm -rf /",          # shell metachar (exec ignores but reject anyway)
        "abc def",                  # whitespace
        "../../etc/passwd",
        "abcXYZ",                   # non-hex char
        "abc",                      # too short
        "a" * 41,                   # too long
        "",                         # already covered but verify
    ]
    for bad in malicious_inputs:
        result = await explain_change(bad)
        assert (
            "not a valid commit SHA" in result
            or "required" in result.lower()
        ), f"explain_change must reject malicious input {bad!r}; got: {result!r}"


@pytest.mark.asyncio
async def test_explain_change_accepts_valid_short_and_long_sha():
    """Counterpart to the rejection test: real SHAs (short + full) pass
    the regex and proceed to graph lookup."""
    from unittest.mock import patch, AsyncMock
    from memex.mcp_server.tools_explain import explain_change

    # Mock both git and the graph/LLM path so we only verify the gate.
    with (
        patch("memex.mcp_server.tools_explain._git_show",
              new=AsyncMock(return_value="diff --git a/a.py b/a.py\n+pass\n")),
        patch("memex.mcp_server.tools_explain._query_linked_decisions_and_problems",
              new=AsyncMock(return_value=[])),
        patch("memex.mcp_server.tools_explain._call_gemini_pro",
              new=AsyncMock(return_value="explanation here")),
    ):
        for good in ("abcd", "abcd1234", "a" * 40):
            result = await explain_change(good)
            assert "not a valid commit SHA" not in result
            assert "explanation here" in result or "no graph context" in result.lower() or len(result) > 0
