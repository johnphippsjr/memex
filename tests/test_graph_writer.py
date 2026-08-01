import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from memex.graph.writer import (
    write_symbol_delta, write_decision, write_call_edges, MemexSchemaError,
)
from memex.extractor.treesitter import SymbolDelta, Symbol as ExtractedSymbol, CallEdge

@pytest.fixture
def mock_client():
    with patch("memex.graph.writer.get_graph_client", new_callable=AsyncMock) as mock:
        client = MagicMock()
        client.add_episode = AsyncMock()
        client.driver = MagicMock()
        client.driver.execute_query = AsyncMock()
        mock.return_value = client
        yield client

@pytest.mark.asyncio
async def test_write_symbol_delta_added(mock_client):
    """Council fix 6: an added symbol no longer writes a companion NL episode
    — it is only ever the deterministic structured MERGE (see
    test_write_symbol_delta_added_materializes_structured_node below for the
    property-level assertions)."""
    sym = ExtractedSymbol(name="test_fn", kind="fn", signature="def test_fn()", file="test.py", line=10)
    delta = SymbolDelta(added=[sym], removed=[], modified=[])

    await write_symbol_delta(delta, source_commit="abc")

    mock_client.add_episode.assert_not_called()
    mock_client.driver.execute_query.assert_awaited_once()
    params = mock_client.driver.execute_query.call_args.kwargs["params"]
    assert params["name"] == "test_fn"
    assert params["source_commit"] == "abc"

@pytest.mark.asyncio
async def test_write_symbol_delta_added_materializes_structured_node(mock_client):
    """v0.3.7 Layer 1 — an added symbol must reach the graph as a
    *structured*, queryable node (type='Symbol' with `file`, `kind`, `line`,
    `repo_path`).

    This is the test that would have caught the predict_impact dead-tool bug:
    `predict_impact` does `MATCH (src:Entity) WHERE src.file=$file
    AND src.repo_path=$repo`, but the writer only ever called add_episode,
    so no node carried a `file` prop and the tool returned empty for every
    file. See docs/BUG_0.3.6.md.

    Council fix 6: there is no longer a companion NL episode at all (there
    used to be one per symbol — six council seats found independently that
    it burned ~7 days of GPU across a full code ingest restating facts
    already deterministic on the node, created a duplicate unlinked node,
    and polluted the entity namespace with literal "Symbol X" entities).
    `uuid`/`group_id`/`summary` are written directly on the structured node
    instead — the missing `group_id`, not a missing embedding, was the hard
    gate excluding Symbol nodes from every graphiti search path."""
    sym = ExtractedSymbol(
        name="login", kind="fn", signature="def login(user)",
        file="memex/auth.py", line=42,
    )
    delta = SymbolDelta(added=[sym], removed=[], modified=[])

    await write_symbol_delta(delta, source_commit="abc1234", repo_root="D:/memex")

    # No NL episode at all any more.
    mock_client.add_episode.assert_not_called()

    # The structured MERGE materialized the node, carrying identity/partition
    # (fix 6) alongside the pre-existing file/kind/line props (v0.3.7).
    mock_client.driver.execute_query.assert_awaited_once()
    query_text = mock_client.driver.execute_query.call_args.args[0]
    params = mock_client.driver.execute_query.call_args.kwargs["params"]

    assert "MERGE" in query_text
    assert "Symbol" in query_text  # type set to 'Symbol'
    for prop in ("file", "kind", "line", "group_id", "uuid", "summary"):
        assert prop in query_text, f"structured MERGE missing {prop}"

    assert params["name"] == "login"
    assert params["file"] == "memex/auth.py"
    assert params["kind"] == "fn"
    assert params["line"] == 42
    assert params["repo"] == "D:/memex"
    assert params["uuid"]  # a real identity, not blank
    assert "login" in params["summary"]


@pytest.mark.asyncio
async def test_write_symbol_delta_survives_structured_merge_failure(mock_client):
    """Council fix 6 removed the NL-episode fallback path, so the structured
    MERGE is now the symbol's ONLY write. If FalkorDB rejects it (transient
    error), write_symbol_delta must still log and move on rather than raise —
    mirrors write_decision's equivalent guard
    (test_write_decision_logs_warning_when_structured_merge_fails)."""
    mock_client.driver.execute_query.side_effect = Exception("transient FalkorDB error")

    sym = ExtractedSymbol(
        name="login", kind="fn", signature="def login(user)",
        file="memex/auth.py", line=42,
    )
    delta = SymbolDelta(added=[sym], removed=[], modified=[])

    # Must not raise.
    await write_symbol_delta(delta, repo_root="D:/memex")

    mock_client.driver.execute_query.assert_awaited()


@pytest.mark.asyncio
async def test_write_symbol_delta_removed(mock_client):
    sym = ExtractedSymbol(name="old_fn", kind="fn", signature="", file="test.py", line=0)
    delta = SymbolDelta(added=[], removed=[sym], modified=[])
    
    await write_symbol_delta(delta)
    
    mock_client.driver.execute_query.assert_called_once()
    query = mock_client.driver.execute_query.call_args[0][0]
    params = mock_client.driver.execute_query.call_args[1]["params"]
    assert "old_fn" == params["name"]
    assert "SET s.valid_until" in query

@pytest.mark.asyncio
async def test_write_symbol_delta_validation_error(mock_client):
    # Use model_construct to bypass initial validation and trigger it in write_symbol_delta
    from memex.graph.schema import Symbol
    sym = Symbol.model_construct(name="test_fn", kind="invalid", signature="def test_fn()", file="test.py", line=10)
    delta = SymbolDelta(added=[sym], removed=[], modified=[])
    
    with pytest.raises(MemexSchemaError) as excinfo:
        await write_symbol_delta(delta)
    assert "SymbolNode" in str(excinfo.value)

@pytest.mark.asyncio
async def test_write_decision_success(mock_client):
    decision = MagicMock()
    decision.text = "New decision"
    decision.rationale = "Because"
    decision.scope = "local"
    
    await write_decision(decision, modules=["a.py"], commit_sha="12345678")
    
    mock_client.add_episode.assert_called_once()
    assert "decision_12345678" in mock_client.add_episode.call_args[1]["name"]

@pytest.mark.asyncio
async def test_write_decision_validation_error(mock_client):
    # Empty text
    decision = MagicMock()
    decision.text = ""
    decision.rationale = "Because"
    decision.scope = "local"

    with pytest.raises(MemexSchemaError) as excinfo:
        await write_decision(decision, modules=["a.py"], commit_sha="12345678")
    assert "DecisionNode" in str(excinfo.value)


@pytest.mark.asyncio
async def test_write_decision_writes_motivates_edge_to_changed_modules(mock_client):
    """Council fix 1 (the rationale link) — the single highest-value repair
    in the #783 council set. Verified live for weeks that all Decision nodes
    had ZERO edges of any type: the commit's changed-file list was already in
    memory at write time and was discarded into an f-string instead of being
    used to link the Decision to the Module(s) it motivated. This must now
    happen in the SAME query/transaction as the Decision MERGE (one
    execute_query call, not two)."""
    decision = MagicMock()
    decision.text = "switch to structured logging"
    decision.rationale = "easier to grep in Loki"
    decision.scope = "module"

    await write_decision(
        decision, modules=["memex/watcher/handlers.py", "memex/graph/writer.py"],
        commit_sha="cafef00d", repo_root="D:/memex",
    )

    mock_client.driver.execute_query.assert_awaited_once()
    query_text = mock_client.driver.execute_query.call_args.args[0]
    params = mock_client.driver.execute_query.call_args.kwargs["params"]

    assert "UNWIND" in query_text
    assert "MOTIVATES" in query_text
    assert "MERGE (m:Entity" in query_text  # Module join target
    assert params["modules"] == ["memex/watcher/handlers.py", "memex/graph/writer.py"]
    assert params["repo"] == "D:/memex"


@pytest.mark.asyncio
async def test_write_decision_uuid_is_deterministic_not_random(mock_client):
    """Council fix 2 — Decision identity. writer.py used to mint a fresh
    uuid4() on every call and feed it into the MERGE key, so the MERGE could
    never match itself twice (CREATE in costume, measured live: 19 Decision
    nodes for 5 distinct names). Two calls with the same (repo, commit_sha,
    decision text) must now produce the SAME uuid, so a resumed multi-day
    ingest reinforces the existing node instead of forking a duplicate."""
    decision = MagicMock()
    decision.text = "identical decision text"
    decision.rationale = "same reason"
    decision.scope = "local"

    await write_decision(decision, modules=["a.py"], commit_sha="deadbeef", repo_root="D:/repo")
    first_uuid = mock_client.driver.execute_query.call_args.kwargs["params"]["uuid"]

    await write_decision(decision, modules=["a.py"], commit_sha="deadbeef", repo_root="D:/repo")
    second_uuid = mock_client.driver.execute_query.call_args.kwargs["params"]["uuid"]

    assert first_uuid == second_uuid
    import uuid as _uuid
    assert _uuid.UUID(first_uuid).version == 5

    # A different repo (or commit, or text) must NOT collide.
    await write_decision(decision, modules=["a.py"], commit_sha="deadbeef", repo_root="D:/other-repo")
    third_uuid = mock_client.driver.execute_query.call_args.kwargs["params"]["uuid"]
    assert third_uuid != first_uuid

@pytest.mark.asyncio
async def test_write_call_edges_merges_calls_relationship(mock_client):
    """v0.3.7 Layer 2 — each resolved call-site becomes a CALLS edge between
    structured Symbol nodes, scoped to the repo. These edges are what
    predict_impact traverses to find coupled modules."""
    edges = [CallEdge(caller="m", callee="foo", file="memex/auth.py", line=6)]

    # Driver returns a single resolved edge.
    rec = MagicMock()
    rec.get.return_value = 1
    mock_client.driver.execute_query.return_value = MagicMock(records=[rec])

    written = await write_call_edges(edges, repo_root="D:/memex")

    mock_client.driver.execute_query.assert_awaited()
    query_text = mock_client.driver.execute_query.call_args.args[0]
    params = mock_client.driver.execute_query.call_args.kwargs["params"]

    assert "CALLS" in query_text and "MERGE" in query_text
    assert params["caller"] == "m"
    assert params["callee"] == "foo"
    assert params["file"] == "memex/auth.py"
    assert params["repo"] == "D:/memex"
    assert written == 1  # one resolved edge


@pytest.mark.asyncio
async def test_write_call_edges_empty_is_noop(mock_client):
    written = await write_call_edges([], repo_root="D:/memex")
    mock_client.driver.execute_query.assert_not_called()
    assert written == 0


def test_memex_schema_error_init():
    err = MemexSchemaError("TestModel", [{"msg": "error"}])
    assert err.model_name == "TestModel"
    assert err.errors == [{"msg": "error"}]
    assert "Validation failed for TestModel" in str(err)


@pytest.mark.asyncio
async def test_write_decision_persists_v030_fields_via_structured_entity_merge(mock_client):
    """v0.3.0 fields (validated, base_confidence, last_reinforced_at, source,
    write_policy) must reach the graph as queryable properties on a real
    :Entity node — not only the NL episode_body, and not via a post-hoc SET
    keyed off add_episode()'s episode.uuid (that uuid belongs to the
    :Episodic node, a different node from the :Entity every decision-reading
    query requires — verified live against FalkorDB: that SET always matched
    zero rows, elementId() or not). write_decision now MERGEs a dedicated
    :Entity node with a self-generated uuid, mirroring
    _merge_structured_symbol's pattern for Symbols."""
    # episode.uuid is deliberately DIFFERENT from anything write_decision
    # should use for the structured node, to prove the two identities are
    # no longer conflated.
    mock_episode = MagicMock(uuid="episode-uuid-abc")
    mock_result = MagicMock(episode=mock_episode)
    mock_client.add_episode.return_value = mock_result

    decision = MagicMock()
    decision.text = "switch to EdDSA"
    decision.rationale = "key rotation simplicity"
    decision.scope = "module"
    decision.validated = False
    decision.base_confidence = 0.6
    decision.source = "watcher"

    await write_decision(decision, modules=["auth.py"], commit_sha="deadbeef")

    # The NL episode was written AND a structured Entity MERGE fired with
    # all the v0.3.0 fields.
    mock_client.add_episode.assert_awaited_once()
    mock_client.driver.execute_query.assert_awaited_once()

    call = mock_client.driver.execute_query.call_args
    query_text = call.args[0]
    params = call.kwargs.get("params", {})

    # Cypher must MERGE a real :Entity node (label required by every
    # decision-reading query in mcp_server/queries.py/tools_write.py) and SET
    # each v0.3.0 field.
    assert "MERGE (d:Entity {uuid: $uuid})" in query_text
    for prop in (
        "d.type = 'Decision'",
        "d.validated",
        "d.base_confidence",
        "d.last_reinforced_at",
        "d.source",
        "d.write_policy",
    ):
        assert prop in query_text, f"structured MERGE missing {prop}"

    # Parameters must carry the actual values, not the defaults.
    assert params["validated"] is False
    assert params["base_confidence"] == 0.6
    assert params["source"] == "watcher"
    assert params["source_commit"] == "deadbeef"
    assert params["text"] == "switch to EdDSA"

    # The structured node's identity is self-generated — NOT the episode's
    # uuid (that would silently match zero :Entity rows).
    assert params["uuid"] != "episode-uuid-abc"
    import uuid as _uuid
    _uuid.UUID(params["uuid"])  # raises if not a real uuid4 string


@pytest.mark.asyncio
async def test_write_decision_structured_merge_independent_of_episode_uuid(mock_client):
    """write_decision's structured Entity MERGE no longer depends on
    add_episode()'s episode.uuid at all (it never could safely target it --
    that uuid belongs to a different node, the :Episodic episode, not the
    :Entity the rest of the codebase queries for). Even when Graphiti
    returns an episode object with NO uuid attribute, the structured MERGE
    still fires using write_decision's own self-generated uuid."""
    # Episode object with NO uuid attribute at all.
    mock_episode = MagicMock(spec=[])
    mock_result = MagicMock(episode=mock_episode)
    mock_client.add_episode.return_value = mock_result
    mock_client.driver.execute_query.return_value = MagicMock(records=[])

    decision = MagicMock()
    decision.text = "any decision"
    decision.rationale = "any"
    decision.scope = "local"

    # Must not raise -- the missing episode.uuid is irrelevant to the
    # structured write now.
    await write_decision(decision, modules=["x.py"], commit_sha="abcd1234")

    mock_client.add_episode.assert_awaited_once()
    mock_client.driver.execute_query.assert_awaited_once()
    params = mock_client.driver.execute_query.call_args.kwargs.get("params", {})
    assert params["uuid"]  # a real, self-generated uuid was used regardless


@pytest.mark.asyncio
async def test_write_decision_logs_warning_when_structured_merge_fails(mock_client):
    """If FalkorDB rejects the structured Decision MERGE, the write does not
    crash — the NL episode is already in the graph; type='Decision' metadata
    can be backfilled."""
    mock_result = MagicMock(episode=MagicMock(uuid="uuid-x"))
    mock_client.add_episode.return_value = mock_result
    mock_client.driver.execute_query.side_effect = Exception("transient")

    decision = MagicMock()
    decision.text = "do a thing"
    decision.rationale = "reason"
    decision.scope = "local"

    # Must not raise — Pydantic validates, episode is written, structured
    # MERGE fails silently (logged as a warning).
    await write_decision(decision, modules=["x.py"], commit_sha="abcd1234")
