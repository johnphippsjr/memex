"""Board #800 — the FalkorDB temporal upper-bound (range-index pushdown) fix.

FalkorDB's RANGE index silently drops `<`/`<=` upper-bound predicates on an
indexed property (valid_at/created_at/expired_at/invalid_at are ISO strings with
a RANGE index), so every temporal upper-bound filter graphiti builds returns the
whole non-null population. Verified live on the test FalkorDB (scratch graph):
`WHERE t <= X` -> 120 of 120 (true 30); the council's WITH-barrier also -> 120;
`toString(t) <= X` -> 30 (correct). These DB-free tests assert both patch sites
emit the toString() form; the end-to-end proof (patched retrieve_episodes
returns the 3 latest episodes BEFORE a datetime reference_time, not the 3 latest
overall) was run live against memex-fork-test — see the board note.
"""

import pytest


def _op(value):
    from graphiti_core.search.search_filters import ComparisonOperator
    return next(o for o in ComparisonOperator if o.value == value)


def test_date_filter_constructor_wraps_comparisons_in_tostring():
    from memex.graph.graphiti_query_patches import apply_all_patches
    from graphiti_core.search import search_filters as sf

    apply_all_patches()  # idempotent

    for symbol in ("<=", "<", ">", ">="):
        q = sf.date_filter_query_constructor("e.valid_at", "$p", _op(symbol))
        assert "toString(e.valid_at)" in q, f"{symbol}: {q}"
        assert symbol in q
        assert "$p" in q


def test_date_filter_constructor_leaves_is_null_alone():
    from memex.graph.graphiti_query_patches import apply_all_patches
    from graphiti_core.search import search_filters as sf
    from graphiti_core.search.search_filters import ComparisonOperator

    apply_all_patches()
    q = sf.date_filter_query_constructor("e.valid_at", "$p", ComparisonOperator.is_null)
    # IS NULL is not a range predicate — no toString wrap, no bound param.
    assert "toString" not in q
    assert "IS NULL" in q.upper()
    assert "$p" not in q


@pytest.mark.asyncio
async def test_retrieve_episodes_query_is_tostring_wrapped():
    """The patched retrieve_episodes must emit `toString(e.valid_at) <=` in BOTH
    branches (plain + saga), never the bare `e.valid_at <=` the range index
    mis-optimizes. A fake executor captures the query without a DB."""
    from memex.graph.graphiti_query_patches import apply_all_patches
    from graphiti_core.driver.falkordb.operations.episode_node_ops import (
        FalkorEpisodeNodeOperations,
    )
    from datetime import datetime, timezone

    apply_all_patches()

    captured = {}

    class FakeExec:
        async def execute_query(self, q, **kw):
            captured["q"] = q
            return ([], None, None)

    ref = datetime(2023, 6, 15, tzinfo=timezone.utc)

    # plain branch
    await FalkorEpisodeNodeOperations.retrieve_episodes(
        object(), FakeExec(), ref, last_n=3, group_ids=["g"]
    )
    assert "toString(e.valid_at) <= $reference_time" in captured["q"]
    assert "WHERE e.valid_at <=" not in captured["q"]

    # saga branch
    await FalkorEpisodeNodeOperations.retrieve_episodes(
        object(), FakeExec(), ref, last_n=3, group_ids=["g"], saga="s"
    )
    assert "toString(e.valid_at) <= $reference_time" in captured["q"]
    assert "WHERE e.valid_at <=" not in captured["q"]
