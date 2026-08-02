import json
import logging
import re
import asyncio
from typing import List
from pydantic import BaseModel, ValidationError
import openai
from memex.config import get_config
from memex.graph.schema import Decision

logger = logging.getLogger(__name__)

class DecisionSchema(BaseModel):
    text: str
    rationale: str
    scope: str  # local, module, project

class DecisionsResponse(BaseModel):
    decisions: List[DecisionSchema]

# Rendered once and embedded directly in the prompt. Under response_format=
# {"type": "json_object"} the API does not enforce any schema (unlike
# "json_schema"'s constrained decoding), so the model only has the prompt to
# go on -- both the field-level schema and a concrete filled-in example are
# given to make the expected shape unambiguous.
_DECISIONS_JSON_SCHEMA = json.dumps(DecisionsResponse.model_json_schema())
_DECISIONS_EXAMPLE = json.dumps({
    "decisions": [
        {
            "text": "Pinned redis to <8 to avoid the unpinned 8.1.0 HIMPORT regression",
            "rationale": "redis 8.1.0 adds a himport_registry kwarg that FalkorDB's sync client rejects, breaking every connection",
            "scope": "project",
        }
    ]
})


def _strip_code_fence(text: str) -> str:
    """Strip a wrapping ```json ... ``` markdown fence if present.

    Local models served through the gateway (llama-swap/vLLM) commonly wrap
    JSON output in a code fence even under a structured response_format;
    a bare json.loads would otherwise raise on it. No-op when there is none
    (mirrors graphiti_core.llm_client.openai_generic_client's own helper).
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*[ \t]*\r?\n?", "", stripped)
        stripped = re.sub(r"\r?\n?```[ \t]*$", "", stripped)
    return stripped.strip()


async def extract_decisions(
    commit_message: str,
    diff_summary: str,
    commit_sha: str,
) -> List[Decision]:
    """
    Uses the configured LiteLLM gateway model to extract zero or more
    architectural decisions from a commit. Includes rate limit retries and
    trivial commit filtering.
    """
    # Guard against empty/whitespace messages
    if not commit_message or not commit_message.strip():
        return []

    # Trivial commit filter (regex-like logic)
    trivial_prefixes = ("wip", "fix", "typo", "merge", "bump", "fmt", "format", "lint", "style")
    msg_lower = commit_message.lower().strip()
    if any(msg_lower.startswith(pref) for pref in trivial_prefixes) or len(msg_lower.split()) < 2:
        logger.debug("Skipping trivial commit: %s", commit_message)
        return []

    config = get_config()
    client = openai.AsyncOpenAI(base_url=config.litellm_base_url, api_key=config.litellm_api_key)

    prompt = f"""
    Analyze the following git commit message and diff summary.
    Extract zero or more architectural or technical decisions made in this commit.
    A decision is a deliberate choice about how the system is built, not just a bug fix or a description of code changes.

    Commit Message: {commit_message}
    Diff Summary: {diff_summary}

    If the commit is trivial (e.g., typos, formatting, WIP, merging), return an empty list.

    Respond with ONLY a single JSON object (no prose, no markdown code fence) in exactly this shape:

    {_DECISIONS_EXAMPLE}

    The JSON schema it must validate against:

    {_DECISIONS_JSON_SCHEMA}
    """

    # Retry logic with exponential backoff
    for attempt in range(3):
        try:
            # openai.AsyncOpenAI's chat.completions.create is natively async
            # (no asyncio.to_thread needed -- that was only required for the
            # synchronous google-genai SDK this replaces).
            #
            # response_format={"type": "json_object"}, NOT "json_schema": local
            # Qwen3.5-35B served via the gateway's local (llama-swap) backend
            # does not reliably honor a json_schema response_format the way
            # DeepInfra/OpenAI-proper do. json_object (schema + a worked
            # example embedded in the prompt above, validated client-side via
            # Pydantic after json.loads) is the mode our production
            # graphiti-mcp deployment uses against this exact same local model
            # (LLM_STRUCTURED_OUTPUT_MODE=json_object) and is confirmed
            # working end-to-end there. This fork defaults to the local
            # gateway model, so json_object is used unconditionally here
            # rather than branching on provider.
            response = await client.chat.completions.create(
                model=config.litellm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                # enable_thinking:false — the local qwen3.5 models (the #787
                # ingest runs against the local card) route EVERY token into
                # hidden reasoning otherwise and return an empty body, so this
                # direct call would extract zero decisions on every commit
                # (board #773 c6551349, same trap SalvagingLocalClient handles
                # for the add_episode path). A provider that doesn't think
                # (DeepInfra) ignores the extra_body, so it is always safe.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

            content = response.choices[0].message.content or ""
            data = json.loads(_strip_code_fence(content))
            parsed = DecisionsResponse.model_validate(data)

            extracted_decisions = []

            for d in parsed.decisions:
                # v0.3.0 defaults (Phase 8 — Hallucination Mitigation):
                #   validated=False  — watcher-synthesised, must be approved via `memex review`
                #   base_confidence — config-driven (Signal Pillar A): the
                #     synthesiser routes through the same initial-confidence
                #     resolver as agent writes instead of hardcoding 0.6, so a
                #     repo can tune how much it trusts unreviewed synthesis.
                #     harness=None resolves the `default` harness (default 0.6).
                #   source="watcher" — distinguishes from agent-recorded decisions
                # ``last_reinforced_at`` is set to ``created_at`` by the writer so the
                # computed_confidence helper has an anchor on freshly-synthesised nodes.
                extracted_decisions.append(Decision(
                    text=d.text,
                    rationale=d.rationale,
                    scope=d.scope,
                    source_commit=commit_sha,
                    source="watcher",
                    validated=False,
                    base_confidence=config.initial_confidence_for(None),
                ))

            return extracted_decisions

        except (json.JSONDecodeError, ValidationError) as e:
            # Malformed/non-conforming JSON is common in json_object mode
            # (no constrained decoding) -- worth a bounded retry rather than
            # failing the commit outright, same backoff as the rate-limit path.
            logger.warning(
                "LLM response failed JSON parse/schema validation on attempt %d: %s",
                attempt + 1, e,
            )
            if attempt == 2:
                logger.error("Failed to extract decisions after 3 attempts due to invalid JSON.")
                return []
            await asyncio.sleep(1)
            continue

        except Exception as e:
            # Check for rate limit or other retryable errors
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str:
                wait_time = (2 ** attempt) + 1
                logger.warning("LiteLLM gateway rate limit hit. Retrying in %ds...", wait_time)
                await asyncio.sleep(wait_time)
                continue

            logger.error("Failed to extract decisions via LiteLLM gateway", exc_info=True)
            return []

    logger.error("Failed to extract decisions after 3 attempts due to rate limits.")
    return []
