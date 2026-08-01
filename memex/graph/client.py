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
            falkor_driver = FalkorDriver(
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
