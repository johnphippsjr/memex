import json
import logging
import typing
from typing import Optional
from neo4j import EagerResult
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient, DEFAULT_MODEL
from graphiti_core.llm_client.config import LLMConfig, DEFAULT_MAX_TOKENS, ModelSize
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError
from graphiti_core.prompts.models import Message
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from pydantic import BaseModel
import openai as _openai
from memex.config import get_config
from memex.graph.graphiti_query_patches import apply_all_patches

logger = logging.getLogger(__name__)


def _salvage_truncated_json(raw: str):
    """Board #790: repair a JSON string truncated mid-token (max_tokens cut) to
    its largest valid prefix, or None. One walk (string-state + bracket stack),
    cut at the last safe value boundary, drop a dangling comma, append closers.
    The counting-loop failure is a real edge whose episode_indices list ran away;
    this keeps the edge instead of losing the whole extraction. Mirrors the same
    function shipped in the seed (homek8 seed_phase2.py, board #790)."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    stack, in_str, esc = [], False, False
    safe_end = safe_stack = None
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                safe_end, safe_stack = i + 1, list(stack)
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            safe_end, safe_stack = i + 1, list(stack)
        elif ch == ",":
            safe_end, safe_stack = i, list(stack)
        elif ch.isdigit() or ch in ".eE+-":
            safe_end, safe_stack = i + 1, list(stack)
    if safe_end is None or not safe_stack:
        return None
    prefix = raw[:safe_end].rstrip().rstrip(",")
    closers = "".join("}" if b == "{" else "]" for b in reversed(safe_stack))
    try:
        return json.loads(prefix + closers)
    except json.JSONDecodeError:
        return None


class SalvagingLocalClient(OpenAIGenericClient):
    """OpenAIGenericClient for the LOCAL-card ingest (#787), with two additions
    over the stock client, each mirroring what the production seed already does:

    1. ``enable_thinking: false`` via extra_body. The local qwen3.5 models route
       EVERY token into hidden reasoning otherwise and return an empty body
       (confirmed live, board #773 ``c6551349``); the stock client does not set
       it, so it only works against providers that don't think.
    2. Board #790 SALVAGE. On a JSONDecodeError (the counting-loop truncation)
       repair the response to its largest valid prefix and return that, instead
       of raising into graphiti's retry which just re-rolls the same runaway.

    _generate_response is copied from graphiti-core 0.29.3 verbatim, plus the
    extra_body kwarg and the salvage branch — no other behaviour changes.
    """

    async def _generate_response(
        self, messages, response_model=None,
        max_tokens: int = DEFAULT_MAX_TOKENS, model_size: ModelSize = ModelSize.medium,
    ):
        openai_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == "user":
                openai_messages.append({"role": "user", "content": m.content})
            elif m.role == "system":
                openai_messages.append({"role": "system", "content": m.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format=self._build_response_format(response_model),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            result = response.choices[0].message.content or ""
            if not result:
                raise EmptyResponseError("LLM returned an empty response")
            cleaned = self._strip_code_fences(result)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                salvaged = _salvage_truncated_json(cleaned)
                if salvaged is not None:
                    logger.warning(
                        "salvaged a truncated extraction response (%d edge(s)) "
                        "instead of dropping it (board #790)",
                        len(salvaged.get("edges", [])) if isinstance(salvaged, dict) else 0,
                    )
                    return salvaged
                raise
        except _openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f"Error in generating LLM response: {e}")
            raise


class _CompatRecord(dict):
    """A plain-dict row that also satisfies ``neo4j.Record``'s ``.data()``
    accessor.

    FalkorDriver.execute_query already converts each FalkorDB result row into
    a plain ``dict`` (see its source) — but ~20 memex call sites across
    cli_graph.py/cli_review.py/graph/cluster.py/graph/cluster_summary.py/
    graph/stats.py/mcp_server/queries.py/mcp_server/team.py/
    mcp_server/tools_explain.py/mcp_server/graph_query.py/
    memory_tool/projection.py call ``record.data()`` on each row, matching
    real ``neo4j.Record``'s API (a plain ``dict`` has no ``.data()``). This
    thin subclass is a no-op for every existing dict use (indexing, ``.get``,
    iteration all still work unchanged) and additionally answers ``.data()``
    the same way ``neo4j.Record.data()`` does with no args — return every
    field as a plain dict.
    """

    def data(self, *keys) -> dict:
        if keys:
            return {k: self.get(k) for k in keys}
        return dict(self)


class CompatFalkorDriver(FalkorDriver):
    """
    Restores the Neo4jDriver `execute_query(cypher, params={...})` calling
    convention on top of graphiti-core 0.29.3's FalkorDriver.

    Every one of memex's own structured (non-NL) Cypher writes/reads —
    graph/writer.py's Symbol MERGE + CALLS edges, watcher/handlers.py's
    decision corroboration, cli_review.py, the cluster/archive/decay/
    governance passes, and the whole mcp_server query/write-tool layer —
    was written against `graphiti_core.driver.neo4j_driver.Neo4jDriver
    .execute_query`, which special-cases a `params` kwarg:
    `params = kwargs.pop('params', None)`. FalkorDriver.execute_query does
    NOT: its signature is `execute_query(self, cypher_query_, **kwargs)` and
    it treats the whole of `**kwargs` as the flat parameter dict. A caller
    passing `params={"name": ...}` therefore sends FalkorDB a single
    top-level parameter literally named "params" — none of the query's real
    `$name`/`$file`/... placeholders are ever bound, and FalkorDB rejects the
    call with `ResponseError: Missing parameters`, on every single one of
    these ~60 call sites, every time.

    `client.add_episode()` itself is unaffected (graphiti-core's own internal
    driver calls already match each driver's real signature) — which is why
    the earlier OpenAIGenericClient fix alone produced real Episodic/Entity
    nodes but every structured Symbol/CALLS/decision write downstream of it
    kept silently failing. Verified live 2026-08-01: a bare
    `FalkorDriver.execute_query(q, params={"now": ...})` fails with the exact
    "Missing parameters" error surfaced during the fork's real-ingestion
    retest; the same call through this subclass succeeds.

    Restoring the same normalization Neo4jDriver already does here keeps
    every existing memex call site (and the unit tests asserting that exact
    `params=` calling convention) working unmodified.
    """

    async def execute_query(self, cypher_query_, **kwargs):
        params = kwargs.pop("params", None)
        if params:
            kwargs.update(params)
        raw = await super().execute_query(cypher_query_, **kwargs)

        # Second half of the same Neo4jDriver-convention restoration this
        # class exists for: graphiti_core.driver.falkordb_driver.FalkorDriver
        # .execute_query returns a bare ``(records, header, None)`` tuple,
        # NOT graphiti-core's Neo4jDriver.execute_query return type
        # (``neo4j.EagerResult``, a NamedTuple exposing `.records`/
        # `.summary`/`.keys`). Every memex call site that reads a query's
        # result — graph/writer.py's write_call_edges/_get_episode_uuid,
        # graph/cluster.py, graph/cluster_runner.py, graph/cluster_summary.py,
        # graph/stats.py, graph/decay.py, cli_graph.py, mcp_server/queries.py
        # (the whole MCP read-tool surface), mcp_server/team.py,
        # mcp_server/tools_explain.py, mcp_server/tools_impact.py,
        # mcp_server/tools_write.py, mcp_server/graph_query.py,
        # watcher/handlers.py — was written against that Neo4j convention
        # and does ``result.records`` directly. Left unwrapped, EVERY one of
        # those raises ``AttributeError: 'tuple' object has no attribute
        # 'records'`` on FalkorDB: most are caught by a bare
        # ``except Exception`` and logged as a false failure (e.g.
        # write_call_edges's "CALLS edge write failed", which fired on every
        # single edge despite the edge having already been written by the
        # same MERGE), a few propagate straight into MemexQueryError and
        # break the MCP read surface outright. Wrapping FalkorDriver's tuple
        # in a real ``neo4j.EagerResult`` here — the exact same kind of
        # normalization this class already does for the ``params=`` calling
        # convention above — fixes every one of those call sites at once
        # instead of patching each individually. ``neo4j`` itself is an
        # unconditional dependency of graphiti-core (not gated behind the
        # ``[falkordb]`` extra), so this import is always available.
        if raw is None:
            # FalkorDriver.execute_query returns bare None on a benign
            # "index already exists" race (see its source) — treat as zero
            # rows rather than let `.records` raise on NoneType downstream.
            return EagerResult(records=[], summary=None, keys=[])
        if isinstance(raw, tuple):
            records, header, _ = raw
            records = [_CompatRecord(r) for r in (records or [])]
            return EagerResult(records=records, summary=None, keys=header or [])
        return raw


class GraphClient:
    """
    Singleton Graphiti client for memex.
    """
    _instance: Optional[Graphiti] = None

    @classmethod
    async def get_instance(cls) -> Graphiti:
        if cls._instance is None:
            config = get_config()

            # board #781: two query-side graphiti-core defects (BM25
            # fulltext dead due to group_id escaping; HNSW vector indexes
            # built but never queried) - see graphiti_query_patches.py for
            # the full writeup and verification. Must be applied before the
            # driver below runs its first search.
            apply_all_patches()

            # FalkorDB graph driver. `database` is the FalkorDB graph key —
            # this MUST match config.unified_group_id (see config.py comment
            # on falkor_graph) so every add_episode() call's explicit
            # group_id equals the driver's already-configured database and
            # graphiti-core never triggers its silent
            # `self.driver.clone(database=group_id)` re-point.
            falkor_driver = CompatFalkorDriver(
                host=config.falkor_host,
                port=config.falkor_port,
                database=config.falkor_graph,
            )

            # Configure LLM Client — LiteLLM gateway, OpenAI-compatible.
            #
            # OpenAIGenericClient (NOT graphiti-core's dedicated OpenAIClient) — verified
            # 2026-08-01: OpenAIClient._create_structured_completion() calls OpenAI's
            # Responses API (`client.responses.parse()`), and our gateway -> DeepInfra
            # rejects every one of those calls with HTTP 400 "tools must not be an empty
            # array" (litellm.BadRequestError / DeepinfraException), so add_episode() never
            # persisted a single node. OpenAIGenericClient instead uses the plain
            # chat.completions.create() + response_format={...} path.
            #
            # structured_output_mode=config.llm_structured_output_mode (default
            # "json_object") — OpenAIGenericClient defaults to native "json_schema"
            # constrained decoding, which DeepInfra honors but this fork's default
            # local model (qwen3.5-35b, served via llama-swap on the .133 R9700)
            # does not reliably. Our production graphiti-mcp deployment runs this
            # exact model behind this exact gateway with
            # LLM_STRUCTURED_OUTPUT_MODE=json_object -- confirmed working there,
            # so the same mode is used here by default. Override via
            # LLM_STRUCTURED_OUTPUT_MODE=json_schema for a provider with real
            # constrained decoding (e.g. DeepInfra/OpenAI-proper).
            # temperature=config.llm_temperature (default 0.0) — graphiti-core's
            # DEFAULT_TEMPERATURE is 1 and nothing overrode it before board #788.
            # Sampling at 1 produced a 20% malformed-output failure rate and made
            # identical input return wildly different extractions run to run. See
            # the long note on `llm_temperature` in memex/config.py for the
            # measured numbers. Extraction is not a generative task; do not
            # un-pin this without re-running that measurement.
            llm_config = LLMConfig(
                api_key=config.litellm_api_key,
                base_url=config.litellm_base_url,
                model=config.litellm_model,
                temperature=config.llm_temperature,
            )
            # SalvagingLocalClient, not the stock OpenAIGenericClient: it adds
            # enable_thinking:false (the local qwen models return an empty body
            # otherwise, board #773) and board #790's truncated-response salvage.
            # Both are no-ops against a well-behaved provider, so this is safe for
            # DeepInfra too — it only changes behaviour on an empty/broken body.
            llm_client = SalvagingLocalClient(
                config=llm_config,
                structured_output_mode=config.llm_structured_output_mode,
            )

            # Configure Embedder — same gateway, bge-m3 (MUST match the
            # model + dims the mem0 seed graph was embedded with, or the
            # code graph's vectors live in a different space and will never
            # actually unify with the mem0 nodes).
            embedder_config = OpenAIEmbedderConfig(
                embedding_model=config.embedding_model,
                embedding_dim=config.embedding_dim,
                api_key=config.litellm_api_key,
                base_url=config.litellm_base_url,
            )
            embedder = OpenAIEmbedder(config=embedder_config)

            # Cross-encoder / reranker — LiteLLM gateway, OpenAI-compatible.
            #
            # graphiti-core 0.29.3 defaults to OpenAIRerankerClient() (its own
            # bare AsyncOpenAI() client, which needs a real OPENAI_API_KEY and
            # talks to api.openai.com) whenever `cross_encoder` isn't passed
            # explicitly to Graphiti(). The original port avoided that hard
            # OpenAI dependency by passing a `NoOpCrossEncoder` stub instead —
            # but that meant no query path could ever get real reranking, even
            # once memex's own search layer grows a `search_()` call site that
            # asks for `EdgeReranker.cross_encoder` (graphiti_core/search/
            # search.py only invokes `cross_encoder.rank()` for that reranker
            # kind; NoOpCrossEncoder's uniform 0.5 score made such a call a
            # silent shuffle). We already have an OpenAI-compatible gateway, so
            # point graphiti-core's REAL OpenAIRerankerClient at it (same
            # litellm_base_url/api_key/model as the LLM client above) instead
            # of standing up a second, separate reranker model.
            # Same temperature pin as the extraction client above. Reranking is a
            # scoring task — sampling it just adds jitter to result ordering.
            reranker_config = LLMConfig(
                api_key=config.litellm_api_key,
                base_url=config.litellm_base_url,
                model=config.litellm_model,
                temperature=config.llm_temperature,
            )
            cross_encoder = OpenAIRerankerClient(config=reranker_config)

            # Initialize Graphiti. `graph_driver=` (not `uri=`) is required
            # for a non-Neo4j backend — passing `uri="falkor://..."` is NOT
            # supported by graphiti-core 0.29.x's Graphiti.__init__: when
            # graph_driver is None it unconditionally builds a Neo4jDriver
            # from uri/user/password (see graphiti_core/graphiti.py).
            cls._instance = Graphiti(
                graph_driver=falkor_driver,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
            )
            logger.info(
                "Graphiti client initialized (FalkorDB %s:%s/%s, LiteLLM model %s, structured_output_mode=%s)",
                config.falkor_host,
                config.falkor_port,
                config.falkor_graph,
                config.litellm_model,
                config.llm_structured_output_mode,
            )

        return cls._instance

    @classmethod
    async def reset(cls):
        """Clears the singleton instance and closes the driver."""
        if cls._instance:
            try:
                await cls._instance.driver.close()
            except Exception:
                pass
            cls._instance = None
            logger.info("Graphiti client reset")

async def get_graph_client() -> Graphiti:
    return await GraphClient.get_instance()

async def reset_graph_client():
    await GraphClient.reset()
