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
