"""
Shared graphiti-core FalkorDB query-layer patches (board #781).

This is the memex-fork twin of
cluster/mcp/graphiti-mcp-official/src/services/graphiti_query_patches.py in
the homek8 repo - kept in sync by hand (the two codebases cannot share a
Python import across repos/deployments), so if you change one, change the
other. Two independent, query-side defects in graphiti-core==0.29.3 were
found by the #780 nine-seat council and INDEPENDENTLY VERIFIED against the
live production graph (mem0-seed-local-verify, read-only via
GRAPH.RO_QUERY, never a write) on 2026-08-01.

DEFECT 1 - BM25/fulltext arm dead since the graph's creation.
`_escape_fulltext_group_id` (graphiti_core/driver/falkordb/fulltext.py:53-55)
backslash-escapes every non-alphanumeric character in the group_id -
including the hyphens in "mem0-seed-local-verify" - and asks RediSearch for
a `@group_id:"mem0\\-seed\\-local\\-verify"` token its indexer never
produced. Verified LIVE: the query graphiti-core actually generates today
returns 0 rows for every probe tried; the identical query with the
@group_id clause removed returns real rows - 13294/186/261 across 3 probes
(episode fulltext "mem0|upstream|migration" and "Neo4j|decision", edge
fulltext "CALLS" - see board #781 mem0 write-up for the exact reproduction).
`EpisodeSearchMethod` has exactly one member (bm25), so this was a TOTAL,
SILENT outage of episode retrieval - all migrated memories unreachable via
graphiti search since the graph's creation. RRF (`1/(rank+1)` summed across
channels) made an empty channel arithmetically indistinguishable from one
never consulted, which is why nothing ever surfaced this. This also affects
this fork's own graph the moment its group_id contains a hyphen (memex's
default group ids do), so it is not just a mem0-seed-local-verify problem.

FIX: delete the RediSearch @group_id clause outright, do not repair the
escaping. graphiti-core's Cypher layer already applies
`{n,e}.group_id IN $group_ids` AFTER the RediSearch stage in every caller
(search_ops.py's node_fulltext_search/edge_fulltext_search/
episode_fulltext_search/community_fulltext_search, search_utils.py:218),
so under one group_id the RediSearch clause has zero selectivity by
construction. Repairing the escaping would only re-add a redundant, slower
filter.

DEFECT 2 - HNSW vector indexes built, maintained, never queried.
graph_queries.get_vector_cosine_func_query() emits a brute-force
`vec.cosineDistance(...)` MATCH-and-scan for every node/edge similarity
search, ignoring the HNSW indexes FalkorDB already builds and maintains on
every write (Entity.name_embedding, RELATES_TO.fact_embedding). Verified
LIVE (read-only, 3 real queries against mem0-seed-local-verify): indexed
db.idx.vector.queryNodes/queryRelationships return the IDENTICAL top-k
ordering as the brute-force scan (uuid-for-uuid, same scores once
converted - see below), 13-42x faster (measured 19.1ms/17.2ms/80.5ms
brute-force vs 1.2ms/1.3ms/1.9ms indexed on 3 real queries).

*** SCORE SEMANTICS - VERIFIED LIVE, NOT ASSUMED, AND THE OPPOSITE OF ***
*** WHAT GENERIC FALKORDB DOCS SUGGEST:                              ***
`db.idx.vector.queryNodes`/`queryRelationships`'s `score` output is the raw
cosine DISTANCE (0 = identical vectors, increasing = less similar) - NOT a
similarity value. Confirmed empirically here by matching a node/edge
against its own embedding and observing score ~= 0 (e.g. 1.19e-07), with
the ranking then increasing monotonically for less-similar neighbours.
graphiti-core's brute-force expression (graph_queries.py's
get_vector_cosine_func_query) instead returns `(2 - cosineDistance) / 2` -
a similarity on the scale `min_score` thresholds and every caller already
expect (1 = identical, descending, `WHERE score > min_score` keeps the
GOOD matches). This module converts on every call so callers cannot tell
the difference from the unpatched brute-force path except in latency.

^^^ THIS CONVERSION IS THE PART THE EXISTING "PROVEN" PATCH IN
cluster/mcp/graphiti-bridge/seed_phase2.py (homek8 repo) DOES NOT DO. That
script's _apply_vector_index_patch() filters/orders on the RAW distance
value directly (`WHERE score > $min_score ... ORDER BY score DESC`), with
no (2-d)/2 conversion. Verified LIVE (read-only) that this is a real,
currently-active bug there: graphiti-core's own node-dedup and
edge-invalidation-candidate callers pass min_score=0.6
(NODE_DEDUP_COSINE_MIN_SCORE / DEFAULT_MIN_SCORE, and
EDGE_HYBRID_SEARCH_RRF's default sim_min_score), and running that exact
patched Cypher with min_score=0.6 returns ZERO rows for a node whose true
nearest neighbours all sit at distance <0.4 (confirmed with the SAME
corrected Cypher below returning the expected, correctly-ordered,
correctly-thresholded candidates for the identical inputs). This module
intentionally does NOT reuse that Cypher as-is; only the *idea* (route
through db.idx.vector.query{Nodes,Relationships}) is lifted, with the
scoring bug fixed. See board #781's mem0 write-up for the full finding
(that bug lives in the OTHER repo's seed script, not here, and the running
graphiti-seed-full job was left untouched per the board #781 hard
constraint).

Apply both via apply_all_patches() as early as possible - before any
Graphiti/FalkorDriver object is constructed - so every code path that goes
through FalkorSearchOperations sees the fix.
"""

from __future__ import annotations

import logging

logger = logging.getLogger('graphiti_query_patches')

_applied = False


def _patch_fulltext_group_clause() -> None:
    """DEFECT 1 fix - delete (not repair) the RediSearch @group_id clause."""
    from graphiti_core.driver.falkordb import fulltext as fulltext_mod
    from graphiti_core.driver.falkordb.operations import search_ops as search_ops_mod
    from graphiti_core.helpers import validate_group_ids

    max_query_length = fulltext_mod.MAX_QUERY_LENGTH
    stopwords = fulltext_mod.STOPWORDS
    sanitize_falkor_fulltext_query = fulltext_mod.sanitize_falkor_fulltext_query

    def build_falkor_fulltext_query_no_group_clause(
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = max_query_length,
    ) -> str:
        """Same as graphiti-core's original function EXCEPT it never builds
        a `@group_id:...` RediSearch clause. group_ids is still validated
        (parity with the original signature on bad input) but is otherwise
        unused here - group scoping happens entirely in the Cypher WHERE
        clause each caller already appends afterwards."""
        validate_group_ids(group_ids)

        filtered_words = [
            word
            for word in sanitize_falkor_fulltext_query(query).split()
            if word.lower() not in stopwords
        ]
        if not filtered_words:
            return ''

        sanitized_query = ' | '.join(filtered_words)
        if len(sanitized_query.split(' ')) >= max_query_length:
            return ''

        return f'({sanitized_query})'

    # Reassign in BOTH module namespaces. search_ops.py did
    # `from graphiti_core.driver.falkordb.fulltext import build_falkor_fulltext_query`
    # at import time, which binds search_ops's OWN module-global name to the
    # (old) function object - patching fulltext_mod's attribute alone would
    # never be seen by search_ops's already-bound name - every call site
    # there (_build_falkor_fulltext_query, build_fulltext_query) looks up
    # the unqualified name `build_falkor_fulltext_query` against
    # search_ops's own module globals at call time. Both names must be
    # reassigned for every call site to see the fix.
    fulltext_mod.build_falkor_fulltext_query = build_falkor_fulltext_query_no_group_clause
    search_ops_mod.build_falkor_fulltext_query = build_falkor_fulltext_query_no_group_clause
    logger.info(
        'graphiti_query_patches: fulltext @group_id clause DELETED (defect #1 - '
        'RediSearch escaping made it unmatchable; Cypher-level group_id IN '
        '$group_ids already scopes every caller)'
    )


def _patch_vector_index_search() -> None:
    """DEFECT 2 fix - route node/edge similarity search through FalkorDB's
    HNSW indexes instead of a brute-force MATCH + vec.cosineDistance scan.
    Verified byte-identical top-k ordering vs brute-force on 3 real queries
    against the live production graph (board #781), with a distance ->
    similarity score conversion the sibling patch in seed_phase2.py is
    missing (see module docstring). Falls back to the original brute-force
    method on ANY exception, or on any call this simplified fast path can't
    handle - fails safe, never fails closed into a dropped/incorrect
    result."""
    from graphiti_core.driver.driver import GraphProvider
    from graphiti_core.driver.falkordb.operations.search_ops import FalkorSearchOperations
    from graphiti_core.driver.record_parsers import (
        entity_edge_from_record,
        entity_node_from_record,
    )
    from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
    from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
    from graphiti_core.search.search_filters import SearchFilters

    orig_node_sim = FalkorSearchOperations.node_similarity_search
    orig_edge_sim = FalkorSearchOperations.edge_similarity_search
    plain_filter = SearchFilters()

    async def indexed_node_similarity_search(
        self, executor, search_vector, search_filter, group_ids=None, limit=10, min_score=0.6
    ):
        if search_filter != plain_filter or not group_ids:
            return await orig_node_sim(
                self, executor, search_vector, search_filter, group_ids, limit, min_score
            )
        cypher = (
            "CALL db.idx.vector.queryNodes('Entity', 'name_embedding', $k, vecf32($search_vector)) "
            'YIELD node AS n, score '
            'WITH n, (2 - score) / 2 AS score '  # distance -> similarity, SAME scale as brute-force
            'WHERE n.group_id IN $group_ids AND score > $min_score '
            'RETURN ' + get_entity_node_return_query(GraphProvider.FALKORDB)
            + ' ORDER BY score DESC LIMIT $limit'
        )
        try:
            records, _, _ = await executor.execute_query(
                cypher,
                k=max(limit * 3, 60),
                search_vector=search_vector,
                group_ids=group_ids,
                min_score=min_score,
                limit=limit,
            )
        except Exception as e:  # noqa: BLE001 - fail SAFE to the proven brute-force path
            logger.warning('indexed node_similarity_search failed (%r), falling back', e)
            return await orig_node_sim(
                self, executor, search_vector, search_filter, group_ids, limit, min_score
            )
        return [entity_node_from_record(r) for r in records]

    async def indexed_edge_similarity_search(
        self,
        executor,
        search_vector,
        source_node_uuid,
        target_node_uuid,
        search_filter,
        group_ids=None,
        limit=10,
        min_score=0.6,
    ):
        if (
            search_filter != plain_filter
            or source_node_uuid is not None
            or target_node_uuid is not None
            or not group_ids
        ):
            return await orig_edge_sim(
                self,
                executor,
                search_vector,
                source_node_uuid,
                target_node_uuid,
                search_filter,
                group_ids,
                limit,
                min_score,
            )
        cypher = (
            "CALL db.idx.vector.queryRelationships('RELATES_TO', 'fact_embedding', $k, vecf32($search_vector)) "
            'YIELD relationship AS e, score '
            'WITH e, (2 - score) / 2 AS score '  # distance -> similarity, SAME scale as brute-force
            'WHERE e.group_id IN $group_ids AND score > $min_score '
            'RETURN ' + get_entity_edge_return_query(GraphProvider.FALKORDB)
            + ' ORDER BY score DESC LIMIT $limit'
        )
        try:
            records, _, _ = await executor.execute_query(
                cypher,
                k=max(limit * 3, 60),
                search_vector=search_vector,
                group_ids=group_ids,
                min_score=min_score,
                limit=limit,
            )
        except Exception as e:  # noqa: BLE001 - fail SAFE to the proven brute-force path
            logger.warning('indexed edge_similarity_search failed (%r), falling back', e)
            return await orig_edge_sim(
                self,
                executor,
                search_vector,
                source_node_uuid,
                target_node_uuid,
                search_filter,
                group_ids,
                limit,
                min_score,
            )
        return [entity_edge_from_record(r) for r in records]

    FalkorSearchOperations.node_similarity_search = indexed_node_similarity_search
    FalkorSearchOperations.edge_similarity_search = indexed_edge_similarity_search
    logger.info(
        'graphiti_query_patches: node/edge similarity search now routed through '
        'db.idx.vector.query{Nodes,Relationships} (defect #2 - brute-force scan '
        'was ignoring the maintained HNSW indexes), with the distance->similarity '
        'score conversion the seed_phase2.py sibling patch is missing (see module '
        'docstring)'
    )


def _patch_temporal_upper_bound_filter() -> None:
    """DEFECT 3 (board #800) - FalkorDB's RANGE index silently DROPS `<`/`<=`
    upper-bound predicates on an indexed property, returning the WHOLE non-null
    population instead of the bounded subset. valid_at/created_at/expired_at/
    invalid_at are all stored as ISO-8601 STRINGS carrying a RANGE index, so
    every temporal upper-bound filter graphiti builds is silently ineffective.

    VERIFIED LIVE on the test FalkorDB (2026-08-02, scratch graph, 120 monthly
    timestamps 2021-2030, RANGE index on the property):
        WHERE t <= X  -> 120  (true 30)   <- upper bound DROPPED
        WHERE t <  X  -> 120  (true 29)   <- dropped
        WHERE t >= X  ->  91  (true 91)   <- lower bound WORKS
        unindexed twin t2 <= X -> 30      <- full scan is correct
        WITH-barrier (the council's proposed fix) -> 120  <- FAILS, re-hoisted
        toString(t) <= X -> 30            <- CORRECT, defeats the pushdown
    In the real retrieve_episodes shape (ORDER BY valid_at DESC LIMIT 3), the
    unpatched query returned the 3 LATEST-dated episodes overall (2030-12/11/10,
    i.e. the future) for a 2023 reference_time; the toString form returned
    2023-06/05/04, the correct 3 latest BEFORE the reference_time. toString() is
    a no-op on an already-string value, so it changes only the query PLAN.

    Why it matters here specifically: retrieve_episodes feeds each extraction its
    "previous episodes". Ingesting a 2021 commit into a graph that already holds
    2026-dated mem0 seed episodes would hand those 2026 episodes to the 2021
    extraction as prior context, baking future context into the stored fact
    (only a full re-ingest undoes it). The operator's code-change-over-time
    'as of commit N' queries depend on this filter being real. FalkorDB has no
    native `datetime()` function (verified: 'Unknown function datetime'), so the
    council's 'native datetime types' alternative is not available.

    Fixed in two places:
      1. date_filter_query_constructor - wrap value_name in toString() for every
         comparison operator (covers SearchFilters valid_at/invalid_at/
         created_at/expired_at date filtering, search_filters.py).
      2. FalkorEpisodeNodeOperations.retrieve_episodes - the inline
         `WHERE e.valid_at <= $reference_time` in BOTH branches (saga + plain).

    KEEP THE homek8 TWIN in sync (cluster/mcp/graphiti-mcp-official/src/services/
    graphiti_query_patches.py) - same defect, same graph.
    """
    from graphiti_core.search import search_filters as sf
    from graphiti_core.search.search_filters import ComparisonOperator
    from graphiti_core.driver.falkordb.operations.episode_node_ops import (
        FalkorEpisodeNodeOperations,
    )
    from graphiti_core.models.nodes.node_db_queries import EPISODIC_NODE_RETURN
    from graphiti_core.driver.record_parsers import episodic_node_from_record

    _NULL_OPS = (ComparisonOperator.is_null, ComparisonOperator.is_not_null)

    def date_filter_query_constructor(value_name, param_name, operator):
        # IS NULL / IS NOT NULL are not range predicates and need no wrapping.
        if operator in _NULL_OPS:
            return '(' + value_name + ' ' + operator.value + ')'
        # toString() defeats the range-index pushdown that drops the upper
        # bound (#800). No-op on the ISO strings these properties hold.
        return '(toString(' + value_name + ') ' + operator.value + ' ' + param_name + ')'

    # search_filters.py's own edge/node filter constructors call this by its
    # module-global name at call time, so reassigning the module attribute is
    # enough for them; no other module imports it by name (verified).
    sf.date_filter_query_constructor = date_filter_query_constructor

    async def retrieve_episodes(
        self, executor, reference_time, last_n=3, group_ids=None, source=None, saga=None
    ):
        """Faithful copy of graphiti-core 0.29.3's FalkorDB retrieve_episodes,
        with the ONLY change being `e.valid_at <=` -> `toString(e.valid_at) <=`
        in both branches (board #800). ORDER BY / LIMIT / params unchanged."""
        if saga is not None and group_ids is not None and len(group_ids) > 0:
            source_clause = 'AND e.source = $source' if source else ''
            query = (
                """
                MATCH (s:Saga {name: $saga_name, group_id: $group_id})-[:HAS_EPISODE]->(e:Episodic)
                WHERE toString(e.valid_at) <= $reference_time
                """
                + source_clause
                + """
                RETURN
                """
                + EPISODIC_NODE_RETURN
                + """
                ORDER BY e.valid_at DESC
                LIMIT $num_episodes
                """
            )
            records, _, _ = await executor.execute_query(
                query,
                saga_name=saga,
                group_id=group_ids[0],
                reference_time=reference_time,
                source=source,
                num_episodes=last_n,
            )
        else:
            source_clause = 'AND e.source = $source' if source else ''
            group_clause = 'AND e.group_id IN $group_ids' if group_ids else ''
            query = (
                """
                MATCH (e:Episodic)
                WHERE toString(e.valid_at) <= $reference_time
                """
                + group_clause
                + source_clause
                + """
                RETURN
                """
                + EPISODIC_NODE_RETURN
                + """
                ORDER BY e.valid_at DESC
                LIMIT $num_episodes
                """
            )
            records, _, _ = await executor.execute_query(
                query,
                reference_time=reference_time,
                group_ids=group_ids,
                source=source,
                num_episodes=last_n,
            )
        return [episodic_node_from_record(r) for r in records]

    FalkorEpisodeNodeOperations.retrieve_episodes = retrieve_episodes
    logger.info(
        'graphiti_query_patches: temporal upper-bound filters wrapped in '
        'toString() (defect #3 / board #800 - FalkorDB RANGE index dropped '
        '<=/< on indexed ISO temporals; patched retrieve_episodes + '
        'date_filter_query_constructor)'
    )


def apply_all_patches() -> None:
    """Idempotent - safe to call more than once (e.g. from more than one
    factory/entry point)."""
    global _applied
    if _applied:
        return
    _patch_fulltext_group_clause()
    _patch_vector_index_search()
    _patch_temporal_upper_bound_filter()
    _applied = True
