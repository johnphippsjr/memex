import logging
from typing import Optional, List
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from memex.config import get_config

logger = logging.getLogger(__name__)

class NoOpCrossEncoder(CrossEncoderClient):
    """
    A CrossEncoder that does nothing, to bypass OpenAI requirements in Graphiti.
    """
    async def rank(self, query: str, documents: List[str]) -> List[float]:
        return [0.5] * len(documents)


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
        return await super().execute_query(cypher_query_, **kwargs)


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
            # chat.completions.create() + response_format={"type": "json_schema", ...} path,
            # which DeepInfra accepts — confirmed live by writing a real Episodic+Entity
            # node with embedding to a test graph.
            llm_config = LLMConfig(
                api_key=config.litellm_api_key,
                base_url=config.litellm_base_url,
                model=config.litellm_model,
            )
            llm_client = OpenAIGenericClient(config=llm_config)

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

            # Initialize Graphiti. `graph_driver=` (not `uri=`) is required
            # for a non-Neo4j backend — passing `uri="falkor://..."` is NOT
            # supported by graphiti-core 0.29.x's Graphiti.__init__: when
            # graph_driver is None it unconditionally builds a Neo4jDriver
            # from uri/user/password (see graphiti_core/graphiti.py).
            cls._instance = Graphiti(
                graph_driver=falkor_driver,
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=NoOpCrossEncoder()
            )
            logger.info(
                "Graphiti client initialized (FalkorDB %s:%s/%s, LiteLLM model %s)",
                config.falkor_host,
                config.falkor_port,
                config.falkor_graph,
                config.litellm_model,
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
