"""memex package init.

Historical note (falkordb-litellm fork): this module used to install a
monkey-patch on ``google.genai``'s ``Models``/``AsyncModels`` classes to add
client-side rate-limiting/auto-retry around direct Gemini API calls. The
fork removes ``google-genai`` as a dependency entirely (LLM/embedding calls
now go through the LiteLLM gateway via ``openai.AsyncOpenAI`` — see
``memex/graph/client.py``, ``memex/synthesizer/commit.py``,
``memex/graph/cluster_summary.py`` and ``memex/mcp_server/tools_explain.py``,
each of which retains its own 429/rate-limit retry loop), so the Gemini-
specific patch has no target left to patch and has been removed rather than
left to silently no-op with a confusing warning on every import.
"""
