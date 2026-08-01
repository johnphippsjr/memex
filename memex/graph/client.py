import logging
from typing import Optional
from neo4j import EagerResult
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from memex.config import get_config

logger = logging.getLogger(__name__)


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
            llm_config = LLMConfig(
                api_key=config.litellm_api_key,
                base_url=config.litellm_base_url,
                model=config.litellm_model,
            )
            llm_client = OpenAIGenericClient(
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
            reranker_config = LLMConfig(
                api_key=config.litellm_api_key,
                base_url=config.litellm_base_url,
                model=config.litellm_model,
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
