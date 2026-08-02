import hashlib
import logging
import uuid as uuid_module
from datetime import datetime, UTC
from pydantic import ValidationError
from memex.config import get_config
from memex.graph.client import get_graph_client
from graphiti_core.nodes import EpisodeType
from memex.graph.schema import SymbolNode, DecisionNode, Dependency
from memex.extractor.treesitter import SymbolDelta

logger = logging.getLogger(__name__)

#: Fixed namespace for Decision identity (council fix 2). Any constant UUID
#: works here — it only has to be stable across processes/runs so that
#: uuid5(namespace, key) is reproducible, never that it means anything on
#: its own.
_DECISION_UUID_NAMESPACE = uuid_module.uuid5(
    uuid_module.NAMESPACE_URL, "https://github.com/johnphippsjr/memex/decision"
)


def _decision_identity_slug(text: str) -> str:
    """Short, stable fingerprint of a decision's text — used ONLY as part of
    the uuid5 identity key below, never displayed (that's d.text, fix 3).
    Needed because multiple distinct decisions are routinely synthesized
    from the SAME commit; keying identity on commit_sha alone (the previous
    episode_name scheme) collapsed them all onto one stub name."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

class MemexSchemaError(Exception):
    """Raised when node data fails Pydantic validation."""
    def __init__(self, model_name: str, errors: list):
        self.model_name = model_name
        self.errors = errors
        super().__init__(f"Validation failed for {model_name}: {errors}")


class MemexWriteError(Exception):
    """Raised when node creation fails or cannot be verified."""


#: Structured-symbol MERGE. Mirrors the post-hoc Cypher pattern already used
#: by write_decision / write_lockfile_delta (ARCHITECTURE-v0.3.0 §4 Q1): the
#: NL add_episode call gives Graphiti a search surface, but the *structured*
#: fields predict_impact relies on (`file`, `kind`, `line`, `type='Symbol'`,
#: `repo_path`) are never parsed out of NL prose, so we write them inline.
#: Without this, `predict_impact`'s `MATCH (src:Entity) WHERE src.file=$file`
#: matches nothing and the tool returns empty for every file (BUG_0.3.6.md).
#: uuid/group_id are set ON CREATE ONLY — identity and graph partition must
#: never churn on a re-index. summary is refreshed on every MATCH alongside
#: kind/signature/line so it stays accurate to the symbol's current shape.
#: Council fix 6: these three properties replace the deleted Symbol NL
#: episode (see write_symbol_delta below) — the council found the missing
#: `group_id`, NOT a missing embedding, is the hard gate that excluded every
#: Symbol node from every graphiti search path. Deliberately no `:Symbol`
#: label and no `name_embedding` — the label question is blocked on board
#: #782's one-write test and is out of scope here.
#: Board #786 (schema half): the Symbol node now carries the ``:Symbol`` label
#: alongside ``:Entity`` AND a ``name_embedding`` over a composed retrieval card.
#: Both are trailing UNCONDITIONAL SETs so they apply on create AND on match (a
#: re-index must refresh the embedding — the signature may have changed).
#:
#: WHY ``:Entity:Symbol`` and not ``:Symbol`` alone: #782 proved a multi-label
#: node is covered by the Entity fulltext index, the Entity vector index, AND
#: both ``MATCH (n:Entity)`` and ``MATCH (n:Symbol)``. So the extra label buys a
#: clean ``MATCH (n:Symbol)`` surface for free without dropping out of any index
#: graphiti or predict_impact already uses.
#:
#: WHY embed at all, and why a CARD not the bare name: #786's retrieval
#: experiment (227 real symbols, 30 ground-truth code queries, prod index shape)
#: measured composed-card MRR .908 / recall@10 .967, versus bare-identifier
#: .730 / .900, versus no-embedding .656 / .800. On natural-language queries the
#: card scored .917 while the bare name scored .460 — i.e. embedding just the
#: identifier is WORSE than not embedding, because bge-m3 on a short snake_case
#: token embeds ORTHOGRAPHICALLY, not semantically. A null/absent name_embedding
#: is silently excluded from vector KNN (also #782), so an embedding failure here
#: degrades to "not vector-searchable", never an error.
_SYMBOL_MERGE_QUERY = """
MERGE (s:Entity {name: $name, file: $file, repo_path: $repo})
  ON CREATE SET s.type = 'Symbol',
                s.kind = $kind,
                s.signature = $signature,
                s.line = $line,
                s.valid_from = $now,
                s.valid_until = NULL,
                s.source_commit = $source_commit,
                s.write_policy = 'locked',
                s.access_count = 0,
                s.last_reinforced_at = $now,
                s.uuid = $uuid,
                s.group_id = $group_id,
                s.summary = $summary
  ON MATCH SET  s.type = 'Symbol',
                s.kind = $kind,
                s.signature = $signature,
                s.line = $line,
                s.valid_until = NULL,
                s.source_commit = coalesce($source_commit, s.source_commit),
                s.last_reinforced_at = $now,
                s.summary = $summary
SET s:Symbol
"""

#: Same as above but also writes the composed-card embedding. Used when the
#: embedding succeeded; the plain query above is the fallback when it did not, so
#: a symbol is still MERGEd and queryable structurally even with the embedder
#: down. Kept as two constants rather than string interpolation so a bad param
#: can never smuggle Cypher into the statement.
_SYMBOL_MERGE_QUERY_EMBEDDED = _SYMBOL_MERGE_QUERY + (
    ", s.name_embedding = vecf32($name_embedding)"
)


def _symbol_card(sym, repo_root: str | None) -> str:
    """The composed retrieval card that gets embedded (board #786, Arm C).

    Arm C's winning card in the experiment was
    ``kind + name + file + signature + docstring + callers``. ``SymbolDelta``
    carries kind/name/file/signature/line but NOT docstring or callers, so this
    is the REDUCED card. That is an honest under-build: the experiment measured
    the card while its docstring coverage undershot the corpus (33.5% vs 62.4%)
    and it STILL won, so the reduced card is expected to help; add docstring +
    caller ingredients here if the extractor ever carries them.
    """
    parts = [f"{sym.kind} {sym.name}"]
    if sym.signature and sym.signature != sym.name:
        parts.append(f"signature: {sym.signature}")
    if sym.file:
        parts.append(f"defined in {sym.file}")
    if repo_root:
        parts.append(f"repo {repo_root}")
    return " | ".join(parts)


async def _merge_structured_symbol(
    client, sym, repo_root: str | None, now, source_commit: str | None,
    group_id: str | None = None,
) -> None:
    """Materialize a queryable, vector-searchable Symbol node.

    Council fix 6: this is the SYMBOL'S ONLY WRITE — no companion NL episode
    (see write_symbol_delta's docstring). ``summary`` is a deterministic string
    built from already-known fields, not an LLM restatement, so the structured
    MERGE stays zero-GPU. Board #786 adds ONE embedding call per symbol on top —
    an embedder hop, not an LLM extraction — for the composed card.
    """
    summary = f"{sym.kind} {sym.name} in {sym.file}" + (
        f", line {sym.line}" if sym.line else ""
    )
    params = {
        "name": sym.name,
        "file": sym.file,
        "repo": repo_root,
        "kind": sym.kind,
        "signature": sym.signature,
        "line": sym.line,
        "now": now,
        "source_commit": source_commit,
        "uuid": str(uuid_module.uuid4()),
        "group_id": group_id,
        "summary": summary,
    }

    # Best-effort embedding: a failure here must degrade the symbol to
    # "structurally present but not vector-searchable", never block the MERGE.
    query = _SYMBOL_MERGE_QUERY
    try:
        embedding = await client.embedder.create(_symbol_card(sym, repo_root))
        params["name_embedding"] = embedding
        query = _SYMBOL_MERGE_QUERY_EMBEDDED
    except Exception:
        logger.warning(
            "Symbol card embedding failed for %s in %s; node will be written "
            "without name_embedding (not vector-searchable until re-index)",
            sym.name, sym.file, exc_info=True,
        )

    try:
        await client.driver.execute_query(query, params=params)
    except Exception:
        logger.warning(
            "structured Symbol MERGE failed for %s in %s; predict_impact may "
            "not see this symbol until the next index pass",
            sym.name,
            sym.file,
            exc_info=True,
        )


#: Board #786 (versioning half) — close any OTHER open version of this symbol.
#: "Open" = ``valid_until IS NULL``. We exclude ``valid_from = $now`` so that
#: RE-PROCESSING the same commit (the ingest is resumable by construction, #787)
#: never closes the very version this run is (re-)writing. At most one prior
#: version is open, so this stamps exactly one boundary.
_SYMBOL_CLOSE_OPEN_QUERY = """
MATCH (s:Entity {name: $name, file: $file, repo_path: $repo})
WHERE s.type = 'Symbol' AND s.valid_until IS NULL AND s.valid_from <> $now
SET s.valid_until = $now
"""

#: Board #786 — upsert THE version valid from ``$now``. The MERGE key includes
#: ``valid_from`` so each commit-time state is its OWN node (bi-temporal history)
#: rather than an in-place overwrite, AND so a re-run is idempotent: the second
#: pass MATCHes the node it wrote the first time and only bumps
#: ``last_reinforced_at``. Trailing unconditional label + embedding, same as the
#: overwrite path.
_SYMBOL_VERSION_MERGE_QUERY = """
MERGE (v:Entity {name: $name, file: $file, repo_path: $repo, valid_from: $now})
  ON CREATE SET v.type = 'Symbol',
                v.kind = $kind,
                v.signature = $signature,
                v.line = $line,
                v.valid_until = NULL,
                v.source_commit = $source_commit,
                v.write_policy = 'locked',
                v.access_count = 0,
                v.last_reinforced_at = $now,
                v.uuid = $uuid,
                v.group_id = $group_id,
                v.summary = $summary
  ON MATCH SET  v.last_reinforced_at = $now
SET v:Symbol
"""
_SYMBOL_VERSION_MERGE_QUERY_EMBEDDED = _SYMBOL_VERSION_MERGE_QUERY + (
    ", v.name_embedding = vecf32($name_embedding)"
)


async def _version_structured_symbol(
    client, sym, repo_root: str | None, now, source_commit: str | None,
    group_id: str | None = None,
) -> None:
    """Append a bi-temporal VERSION of a Symbol instead of overwriting it.

    Used only by the historical ingest (#787), where each commit records what a
    symbol looked like *at that commit*. The live watcher keeps using
    ``_merge_structured_symbol`` (overwrite-in-place) — you do not want a new
    version node per editor save.

    Two steps, in order:
      1. Close any OTHER currently-open version (stamp ``valid_until = now``).
      2. Upsert the node whose ``valid_from = now``, idempotently.

    A symbol re-observed at a later commit with the SAME signature would, under
    the delta-driven ingest, not appear in ``added``/``modified`` at all — so
    this is only reached for genuine first-sightings and real changes. The
    idempotent MERGE key means a resumed run re-writing the same commit is a
    no-op, not a duplicate version.
    """
    summary = f"{sym.kind} {sym.name} in {sym.file}" + (
        f", line {sym.line}" if sym.line else ""
    )
    params = {
        "name": sym.name,
        "file": sym.file,
        "repo": repo_root,
        "kind": sym.kind,
        "signature": sym.signature,
        "line": sym.line,
        "now": now,
        "source_commit": source_commit,
        "uuid": str(uuid_module.uuid4()),
        "group_id": group_id,
        "summary": summary,
    }

    query = _SYMBOL_VERSION_MERGE_QUERY
    try:
        params["name_embedding"] = await client.embedder.create(
            _symbol_card(sym, repo_root)
        )
        query = _SYMBOL_VERSION_MERGE_QUERY_EMBEDDED
    except Exception:
        logger.warning(
            "Symbol card embedding failed for %s in %s (versioned write); node "
            "will be written without name_embedding",
            sym.name, sym.file, exc_info=True,
        )

    try:
        await client.driver.execute_query(_SYMBOL_CLOSE_OPEN_QUERY, params={
            "name": sym.name, "file": sym.file, "repo": repo_root, "now": now,
        })
        await client.driver.execute_query(query, params=params)
    except Exception:
        logger.warning(
            "versioned Symbol write failed for %s in %s; history for this "
            "symbol may be incomplete until the next pass",
            sym.name, sym.file, exc_info=True,
        )


async def write_symbol_delta(
    delta: SymbolDelta,
    source_commit: str | None = None,
    repo_root: str | None = None,
    commit_time: datetime | None = None,
    versioned: bool = False,
) -> None:
    """
    Writes a SymbolDelta to Graphiti.

    Each added/modified symbol is written ONCE, as a deterministic structured
    MERGE (:Entity {type:'Symbol'}) carrying the queryable ``file``/``kind``/
    ``line``/``repo_path`` props ``predict_impact`` traverses, plus ``uuid``/
    ``group_id``/``summary`` (v0.3.7 Layer 1 + council fix 6).

    Council fix 6 — there is deliberately NO companion NL ``add_episode`` call
    here any more (there used to be one per symbol). Six council seats found
    it independently: it burned roughly seven days of GPU across a full code
    ingest to have an LLM restate facts already deterministic on the node; it
    created a DUPLICATE, unlinked node because Symbol nodes are invisible to
    graphiti's own dedup (no name_embedding); and it polluted the entity
    namespace (13.4% of extracted entities came back literally named
    "Symbol X"). ``group_id`` is written directly on the structured node
    instead — the council established that the missing ``group_id``, not a
    missing embedding, was the hard gate excluding these nodes from every
    graphiti search path.

    Council fix 4 — ``commit_time``, when the caller has a real one (a
    commit's actual date, or the moment a live file-change was detected),
    is used as the node's ``valid_from``/``last_reinforced_at`` anchor
    instead of a fresh ``datetime.now(UTC)`` call. Without this, replaying
    years of git history stamps every symbol with the ingestion wall-clock
    time it happened to be processed at, not the time it was true from.

    Board #786 (versioning) — ``versioned`` selects the write MODE:
      * ``False`` (default, the LIVE WATCHER): overwrite-in-place. A file
        changing on disk should update the symbol's current shape, not spawn a
        version node per editor save.
      * ``True`` (the HISTORICAL INGEST, #787): append a bi-temporal version.
        Each commit records what the symbol was AT that commit — a changed
        signature closes the old version (``valid_until = commit_time``) and
        opens a new one (``valid_from = commit_time``). This is what makes the
        operator's "record changes over time" decision real instead of the
        overwrite that silently discarded it. Requires ``commit_time`` to be the
        real commit date, or every version stamps the wall-clock ingest moment.
    """
    client = await get_graph_client()
    config = get_config()
    now = commit_time or datetime.now(UTC)
    _write = _version_structured_symbol if versioned else _merge_structured_symbol

    # 1. Added symbols
    for sym in delta.added:
        try:
            # Validate
            SymbolNode(
                name=sym.name,
                kind=sym.kind,
                signature=sym.signature,
                file=sym.file,
                line=sym.line,
                valid_from=now,
                source_commit=source_commit
            )
        except ValidationError as e:
            raise MemexSchemaError("SymbolNode", e.errors())

        # Deterministic, LLM-free structured node — the only write for this
        # symbol now (see docstring above). Overwrite or append-version per
        # ``versioned``.
        await _write(
            client, sym, repo_root, now, source_commit,
            group_id=config.unified_group_id,
        )

    # 1b. Modified symbols — overwrite refreshes signature/line in place; the
    # versioned path closes the prior version and opens a new one at commit_time.
    for sym in delta.modified:
        await _write(
            client, sym, repo_root, now, source_commit,
            group_id=config.unified_group_id,
        )

    # 2. Removed symbols — close the OPEN version only. Board #786: the old
    # query set valid_until on EVERY matching node, which in versioned mode
    # would overwrite the historical close-date of already-closed versions.
    # Scoping to `valid_until IS NULL` closes exactly the currently-live version
    # and is correct for the overwrite path too (it has a single open node).
    for sym in delta.removed:
        # Board #802: scope removal to the SAME repo. The creating MERGE is
        # keyed on {name,file,repo_path}, but this removal matched only
        # {name,file} — and the operator chose ONE graph (per-repo partitioning
        # rejected), the exact config where main.py/__init__.py/cli.py/setup.py
        # collide across repos. Without the repo_path predicate, deleting `run`
        # from repo A's app/main.py silently stamps valid_until on repo B's
        # identically-pathed `run` too — a silent write, never an error. The
        # #788 infra symbols (path-free identity Kind/[ns]/name) reuse this same
        # loop, so the same predicate protects them. coalesce(...,'') on BOTH
        # sides, not a bare ``s.repo_path = $repo``: repo_root is None for the
        # live watcher's single-repo case, and a bare ``= NULL`` is never true in
        # Cypher — that would silently STOP closing removed symbols there. The
        # coalesce makes None match None (one unscoped scope) while keeping two
        # real, distinct repo_paths apart, which is the whole point of the gate.
        query = """
        MATCH (s:Entity {name: $name})
        WHERE (s.type = 'Symbol' OR s.name CONTAINS 'Symbol')
              AND s.file = $file
              AND coalesce(s.repo_path, '') = coalesce($repo, '')
              AND s.valid_until IS NULL
        SET s.valid_until = $now
        """
        await client.driver.execute_query(query, params={
            "name": sym.name,
            "file": sym.file,
            "repo": repo_root,
            "now": now
        })

    return {
        "symbols": len(delta.added) + len(delta.modified),
        # No NL episode is attempted for symbols any more (fix 6), so this is
        # always 0. Kept in the summary dict so callers/health.record's
        # existing key doesn't disappear out from under them.
        "episodes_skipped": 0,
    }

#: CALLS-edge MERGE. Resolution is deliberately CONSERVATIVE: a call-site's
#: callee name is linked only when it resolves to exactly ONE structured Symbol
#: node in the repo (``size(cs)=1``). Ambiguous names (e.g. a `run` defined in
#: five files) produce no edge in v1 rather than fan-out false coupling. Calls
#: to stdlib/builtins (no Symbol node) naturally produce no edge. This is what
#: predict_impact traverses, so precision beats recall here. (v0.3.7 Layer 2)
_CALL_EDGE_QUERY = """
MATCH (caller:Entity {name: $caller, file: $file, repo_path: $repo})
WHERE caller.type = 'Symbol'
MATCH (callee:Entity {name: $callee, repo_path: $repo})
WHERE callee.type = 'Symbol'
WITH caller, collect(DISTINCT callee) AS cs
WHERE size(cs) = 1
UNWIND cs AS callee
MERGE (caller)-[r:CALLS]->(callee)
  ON CREATE SET r.created_at = $now,
                r.expired_at = NULL,
                r.line = $line,
                r.last_reinforced_at = $now
  ON MATCH SET  r.expired_at = NULL,
                r.line = $line,
                r.last_reinforced_at = $now
RETURN count(r) AS n
"""


async def write_call_edges(calls, repo_root: str | None = None) -> int:
    """Persist CALLS edges for a file's resolved call-sites.

    Expects :class:`memex.extractor.treesitter.CallEdge` items. Returns the
    number of edges written/refreshed. Best-effort per edge — a failure on one
    call-site is logged and does not abort the rest.
    """
    if not calls:
        return 0

    client = await get_graph_client()
    now = datetime.now(UTC)
    written = 0

    for edge in calls:
        # Skip self-recursion: a function calling itself isn't cross-symbol
        # coupling and would just be noise.
        if edge.caller == edge.callee:
            continue
        try:
            res = await client.driver.execute_query(
                _CALL_EDGE_QUERY,
                params={
                    "caller": edge.caller,
                    "callee": edge.callee,
                    "file": edge.file,
                    "repo": repo_root,
                    "now": now,
                    "line": edge.line,
                },
            )
            if res.records:
                written += int(res.records[0].get("n") or 0)
        except Exception:
            logger.warning(
                "CALLS edge write failed for %s -> %s in %s",
                edge.caller,
                edge.callee,
                edge.file,
                exc_info=True,
            )

    return written


#: REFERENCES-edge MERGE for infra resources (board #788 GAP 1). Same
#: CONSERVATIVE resolution as CALLS: a resource->resource reference is linked
#: only when the referent (kind, name) resolves to exactly ONE resource node in
#: the repo (``size(ts)=1``). A referent's identity is ``Kind/[ns/]name``, so we
#: match on the ``Kind/`` prefix + ``/name`` suffix — this resolves across
#: namespaces without needing the referencing manifest to know the target's ns.
#: External references (a container ``image`` from a registry) have no resource
#: node and so naturally produce no edge, which is correct: the graph models the
#: repo's own dependency structure, not the outside world.
_RESOURCE_REF_EDGE_QUERY = """
MATCH (src:Entity {name: $src, repo_path: $repo})
WHERE src.type = 'Symbol' AND src.kind = 'resource'
MATCH (tgt:Entity {repo_path: $repo})
WHERE tgt.type = 'Symbol' AND tgt.kind = 'resource'
      AND tgt.name STARTS WITH $kind_prefix AND tgt.name ENDS WITH $name_suffix
WITH src, collect(DISTINCT tgt) AS ts
WHERE size(ts) = 1
UNWIND ts AS tgt
MERGE (src)-[r:REFERENCES]->(tgt)
  ON CREATE SET r.created_at = $now,
                r.expired_at = NULL,
                r.via = $via,
                r.ref_kind = $ref_kind,
                r.last_reinforced_at = $now
  ON MATCH SET  r.expired_at = NULL,
                r.via = $via,
                r.last_reinforced_at = $now
RETURN count(r) AS n
"""


#: Board #803 — record a reference that DID NOT resolve to exactly one in-repo
#: resource (0 matches = dangling/external, or >1 = ambiguous), instead of
#: dropping it silently as the size(ts)=1 guard did. The placeholder target is a
#: distinct ``type:'UnresolvedRef'`` :Entity (never collides with a real Symbol),
#: and the edge is a REFERENCES edge flagged ``unresolved:true`` — TRAVERSAL-ONLY
#: (no group_id, no fact_embedding), so it stays out of semantic search per the
#: council's "pure structure stays traversal-only". A later ingest (2nd repo) or a
#: report can then SEE the dangling dependency rather than it being invisible.
_UNRESOLVED_REF_QUERY = """
MATCH (src:Entity {name: $src, repo_path: $repo})
WHERE src.type = 'Symbol' AND src.kind = 'resource'
MERGE (u:Entity {name: $refname, repo_path: $repo, type: 'UnresolvedRef'})
  ON CREATE SET u.created_at = $now, u.write_policy = 'locked',
                u.access_count = 0, u.ref_kind = $ref_kind
MERGE (src)-[r:REFERENCES {via: $via}]->(u)
  ON CREATE SET r.created_at = $now, r.unresolved = true,
                r.ref_kind = $ref_kind, r.expired_at = NULL
  ON MATCH SET  r.unresolved = true
RETURN count(r) AS n
"""


async def _record_unresolved_ref(client, src_identity, ref, repo_root, now) -> int:
    """Best-effort UNRESOLVED_REF record for one dangling/ambiguous resource ref."""
    try:
        await client.driver.execute_query(_UNRESOLVED_REF_QUERY, params={
            "src": src_identity,
            "repo": repo_root,
            "refname": f"{ref.kind}/{ref.name}",
            "via": ref.via,
            "ref_kind": ref.kind,
            "now": now,
        })
        return 1
    except Exception:
        logger.warning(
            "UNRESOLVED_REF record failed for %s -> %s/%s in repo %s",
            src_identity, ref.kind, ref.name, repo_root, exc_info=True,
        )
        return 0


async def write_resource_ref_edges(extract, repo_root: str | None = None) -> int:
    """Persist REFERENCES edges for a k8s file's resolved resource references.

    Expects a :class:`memex.extractor.k8s.K8sExtract` (its ``.refs`` maps a
    resource symbol-key to the ResourceRefs it declares). Returns the number of
    edges written/refreshed. Best-effort per edge; mirrors write_call_edges.

    Only references that resolve to exactly one resource node in the same repo
    become edges — Image refs and dangling names (a Secret referenced but not
    defined in this repo) are silently left un-linked rather than fabricating a
    target node.
    """
    refs_by_key = getattr(extract, "refs", None) or {}
    if not refs_by_key:
        return 0

    client = await get_graph_client()
    now = datetime.now(UTC)
    written = 0
    unresolved = 0

    for key, refs in refs_by_key.items():
        # symbol-key is "<identity>:resource"; the identity is the node name.
        src_identity = key.rsplit(":", 1)[0]
        for ref in refs:
            if ref.kind == "Image":
                continue  # external, no in-repo node to point at
            try:
                res = await client.driver.execute_query(
                    _RESOURCE_REF_EDGE_QUERY,
                    params={
                        "src": src_identity,
                        "repo": repo_root,
                        "kind_prefix": f"{ref.kind}/",
                        "name_suffix": f"/{ref.name}",
                        "via": ref.via,
                        "ref_kind": ref.kind,
                        "now": now,
                    },
                )
                n = int(res.records[0].get("n") or 0) if res.records else 0
                written += n
                if n == 0:
                    # Board #803: did NOT resolve to exactly one in-repo resource
                    # (0 = external/dangling, >1 = ambiguous). Record it rather
                    # than dropping it silently.
                    unresolved += await _record_unresolved_ref(
                        client, src_identity, ref, repo_root, now
                    )
            except Exception:
                logger.warning(
                    "REFERENCES edge write failed for %s -> %s/%s in repo %s",
                    src_identity, ref.kind, ref.name, repo_root, exc_info=True,
                )

    if unresolved:
        logger.info(
            "write_resource_ref_edges: %d edge(s) written, %d UNRESOLVED_REF "
            "recorded (repo %s)", written, unresolved, repo_root,
        )
    return written


#: Structured Decision node + THE RATIONALE LINK, in one transaction.
#:
#: Council fix 1 (the rationale link — the operator's primary goal): all 19
#: Decision nodes were verified live to have ZERO edges of any type. The
#: commit's changed-file list (`modules`, from CommitEvent.files_changed) was
#: already in memory at write time and simply discarded into an f-string.
#: This query now MERGEs a MOTIVATES edge from the Decision to a Module
#: Entity for every changed file. The join target — `:Entity {type:'Module',
#: name:<repo-relative path>, repo_path}` — is the SAME node/key
#: write_lockfile_delta's IMPORTS-edge query already MERGEs (writer.py's
#: edge_query below), so whichever subsystem runs first creates the stub and
#: the other reinforces it; there is nothing new to keep in sync.
#: corroborate_decisions (watcher/handlers.py) requires exactly this edge
#: shape for its Pass 1 file-match gate.
#:
#: Council fix 2 (Decision identity): $uuid is now uuid5(repo, commit_sha,
#: text-slug) — deterministic — instead of a fresh uuid4() minted on every
#: call. uuid4-in-a-MERGE-key can never match itself twice (CREATE in
#: costume, measured: 19 Decision nodes for 5 distinct names); uuid5 lets a
#: resumed multi-day ingest MERGE the same node instead of forking a
#: duplicate. The ON MATCH branch below — previously absent entirely — is
#: what makes that MERGE meaningful instead of a no-op.
#:
#: See write_decision's docstring for why this replaced a post-hoc SET keyed
#: off add_episode()'s episode.uuid (that uuid belongs to the :Episodic node,
#: never to an :Entity, so it could never match `MATCH (d:Entity)` — the
#: label every decision-reading query in mcp_server/queries.py and
#: tools_write.py requires).
_DECISION_MERGE_QUERY = """
MERGE (d:Entity {uuid: $uuid})
  ON CREATE SET d.type = 'Decision',
                d.name = $name,
                d.text = $text,
                d.rationale = $rationale,
                d.scope = $scope,
                d.created_at = $now,
                d.last_reinforced_at = $now,
                d.source = $source,
                d.source_commit = $source_commit,
                d.confidence = $confidence,
                d.validated = $validated,
                d.base_confidence = $base_confidence,
                d.write_policy = 'open',
                d.access_count = 0
  ON MATCH SET  d.last_reinforced_at = $now,
                d.access_count = coalesce(d.access_count, 0) + 1,
                d.source_commit = coalesce(d.source_commit, $source_commit)
WITH d
UNWIND $modules AS module_path
MERGE (m:Entity {name: module_path, repo_path: $repo})
  ON CREATE SET m.type = 'Module',
                m.created_at = $now,
                m.write_policy = 'locked',
                m.access_count = 0
MERGE (d)-[r:MOTIVATES]->(m)
  ON CREATE SET r.created_at = $now,
                r.expired_at = NULL
  ON MATCH SET  r.last_reinforced_at = $now,
                r.expired_at = NULL
"""


#: Board #803 — a stable, deterministic uuid for a Module node keyed on
#: (repo_root, module_path). Module nodes are memex-MERGEd (not graphiti-created)
#: and so carry no uuid of their own (schema.py's uuid_or_natural_key notes this);
#: the MOTIVATES searchable shadow below needs both endpoints to have a real uuid
#: so graphiti's edge-return/reconstruction (source_node_uuid/target_node_uuid)
#: can look the endpoints back up. uuid5 keeps it identical across re-ingests.
_MODULE_UUID_NAMESPACE = uuid_module.uuid5(
    uuid_module.NAMESPACE_URL, "https://github.com/johnphippsjr/memex/module"
)


def _module_uuid(repo_root: str | None, module_path: str) -> str:
    return str(uuid_module.uuid5(_MODULE_UUID_NAMESPACE, f"{repo_root or ''}|{module_path}"))


#: Board #803 (council default: MAKE RATIONALE FINDABLE) — a deterministic,
#: SEARCHABLE ``RELATES_TO`` twin of the structured ``MOTIVATES`` edge. graphiti's
#: whole semantic search surface only ever traverses ``[e:RELATES_TO]`` between
#: :Entity nodes (search_utils.py hardcodes it in ~10 places), so the structured
#: MOTIVATES edge - a distinct relationship type - is invisible to search. The
#: council's decision: rationale (MOTIVATES) SHOULD be findable, so write a
#: deterministic RELATES_TO shadow carrying the decision's rationale as the
#: ``fact`` (what BM25 + the vector index actually match), while pure-structure
#: edges (CALLS/REFERENCES) stay traversal-only. The shadow mirrors the exact
#: property set graphiti's own RELATES_TO edges carry (verified live against the
#: test graph: uuid/source_node_uuid/target_node_uuid/name/fact/group_id/
#: created_at/valid_at/expired_at/invalid_at/reference_time/fact_embedding). It
#: is a TWIN of MOTIVATES, not a replacement - the structured MOTIVATES edge
#: still exists for corroborate_decisions' Pass-1 file-match gate.
_MOTIVATES_SHADOW_QUERY = """
MATCH (d:Entity {uuid: $decision_uuid})
MERGE (m:Entity {name: $module_path, repo_path: $repo})
  ON CREATE SET m.type = 'Module', m.created_at = $now,
                m.write_policy = 'locked', m.access_count = 0
SET m.uuid = coalesce(m.uuid, $module_uuid)
MERGE (d)-[r:RELATES_TO {uuid: $edge_uuid}]->(m)
  ON CREATE SET r.name = 'MOTIVATES',
                r.fact = $fact,
                r.group_id = $group_id,
                r.source_node_uuid = $decision_uuid,
                r.target_node_uuid = coalesce(m.uuid, $module_uuid),
                r.created_at = $now,
                r.valid_at = $now,
                r.reference_time = $now,
                r.expired_at = NULL,
                r.invalid_at = NULL,
                r.episodes = []
  ON MATCH SET  r.fact = $fact,
                r.valid_at = $now
"""
_MOTIVATES_SHADOW_QUERY_EMBEDDED = _MOTIVATES_SHADOW_QUERY + (
    ", r.fact_embedding = vecf32($fact_embedding)"
)


async def _write_motivates_shadow(
    client, decision_uuid: str, decision, modules: list[str],
    repo_root: str | None, now, group_id: str | None,
) -> None:
    """Write a searchable RELATES_TO twin of each MOTIVATES edge (board #803).

    Best-effort, exactly like the Symbol card embedding: an embedder failure
    degrades the shadow to BM25-only (``fact`` is still fulltext-indexed), never
    blocks the write. The ``fact`` sentence is what search matches, so it leads
    with the rationale, not a node label.
    """
    text = (getattr(decision, "text", "") or "").strip()
    rationale = (getattr(decision, "rationale", "") or "").strip()
    for module_path in modules:
        fact = f"{text} (rationale: {rationale})" if rationale else text
        fact = f"{fact} — motivates {module_path}"
        params = {
            "decision_uuid": decision_uuid,
            "module_path": module_path,
            "repo": repo_root,
            "module_uuid": _module_uuid(repo_root, module_path),
            "edge_uuid": str(uuid_module.uuid5(
                _DECISION_UUID_NAMESPACE, f"motivates|{decision_uuid}|{module_path}"
            )),
            "fact": fact,
            "group_id": group_id,
            "now": now,
        }
        query = _MOTIVATES_SHADOW_QUERY
        try:
            params["fact_embedding"] = await client.embedder.create(fact)
            query = _MOTIVATES_SHADOW_QUERY_EMBEDDED
        except Exception:
            logger.warning(
                "MOTIVATES shadow embedding failed for decision %s -> %s; shadow "
                "written BM25-only (not vector-searchable until re-index)",
                decision_uuid, module_path, exc_info=True,
            )
        try:
            await client.driver.execute_query(query, params=params)
        except Exception:
            logger.warning(
                "MOTIVATES shadow write failed for decision %s -> %s; rationale "
                "remains traversal-only via the structured MOTIVATES edge",
                decision_uuid, module_path, exc_info=True,
            )


async def write_decision(
    decision,
    modules: list[str],
    commit_sha: str,
    confidence: float = 1.0,
    source: str = "watcher",
    repo_root: str | None = None,
    commit_time: datetime | None = None,
    entity_types: dict | None = None,
    searchable_rationale: bool = False,
) -> None:
    """
    Writes a technical decision to Graphiti, plus its MOTIVATES edges to the
    changed modules (council fix 1) under a deterministic identity (council
    fix 2). ``repo_root`` should be the SAME canonicalized repo path
    write_lockfile_delta's Module nodes use, or the MOTIVATES edge's Module
    endpoint will not be the same node as the one IMPORTS edges point at.
    ``commit_time``, when known (the commit's real authored/committed time —
    see watcher/git_hook.py), anchors this decision's reference_time/
    created_at instead of the ingestion wall-clock time (council fix 4).

    ``entity_types`` (board #787) is passed straight through to the NL
    ``add_episode`` call. The historical ingest passes
    ``{"Symbol": SymbolEntityType}`` so that when this decision's extraction
    resolves an entity onto an existing :Symbol node, graphiti-core's
    ``extract_attributes_from_nodes`` overlay-merge preserves the node's
    ground-truth structural props (file/line/signature/repo_path) instead of
    replacing them with ``{}`` — see SymbolEntityType's docstring. The live
    watcher leaves it ``None`` (unchanged behaviour): its symbols are written
    overwrite-in-place, and no episode there resolves onto them by name.
    """
    client = await get_graph_client()
    config = get_config()
    now = commit_time or datetime.now(UTC)

    # v0.3.0: preserve any v0.3.0 fields set by the synthesizer (validated,
    # base_confidence) and seed last_reinforced_at = created_at so the
    # computed-confidence helper in memex.graph.confidence has an anchor.
    # Use a sentinel-check style so MagicMock instances (which auto-create
    # attributes) fall through to the defaults cleanly.
    def _real_attr(obj, name, default):
        val = getattr(obj, name, default)
        # MagicMock auto-creates attributes; reject anything that isn't a
        # plain JSON-friendly scalar of the expected type.
        if val is default:
            return default
        if isinstance(default, bool) and not isinstance(val, bool):
            return default
        if isinstance(default, (int, float)) and not isinstance(val, (int, float)):
            return default
        if isinstance(default, str) and not isinstance(val, str):
            return default
        return val

    validated = bool(_real_attr(decision, "validated", False))
    base_confidence = float(_real_attr(decision, "base_confidence", confidence))
    decision_source = _real_attr(decision, "source", source) or source

    try:
        # Validate
        DecisionNode(
            text=decision.text,
            rationale=decision.rationale,
            scope=decision.scope,
            created_at=now,
            source_commit=commit_sha,
            confidence=confidence,
            source=decision_source,
            validated=validated,
            base_confidence=base_confidence,
            last_reinforced_at=now,
        )
    except ValidationError as e:
        raise MemexSchemaError("DecisionNode", e.errors())

    episode_name = f"decision_{commit_sha[:8]}"

    # NL episode — best-effort search surface for Graphiti's own
    # search()/embeddings. Failure here does not block the structured write
    # below (mirrors write_symbol_delta's "structured node first, NL episode
    # best-effort" ordering).
    try:
        await client.add_episode(
            name=episode_name,
            episode_body=(
                f"Decision: {decision.text}. Rationale: {decision.rationale}. "
                f"Scope: {decision.scope}. Affected modules: {', '.join(modules)} "
                f"(Confidence: {confidence}, Source: {decision_source}, "
                f"Validated: {validated}, BaseConfidence: {base_confidence})"
            ),
            source_description=f"git commit {commit_sha}",
            reference_time=now,
            # source=text, NOT the EpisodeType.message default. graphiti's
            # message prompt's FIRST rule is "always extract the speaker (the
            # part before the colon)", and this body opens with "Decision: ...",
            # so the default makes the extractor treat the literal word
            # "Decision" as a dialogue participant. Board #788 measured the fix
            # at 1.8x more relationships (20 paired commits to 8, p ~ 0.036) and
            # it halves the class of episode whose only extracted entity is a
            # label or an author name. NOTE: only measurable at temperature 0 -
            # at graphiti's default of 1 the run-to-run variance is larger than
            # the effect and inverts its apparent direction.
            source=EpisodeType.text,
            group_id=config.unified_group_id,
            # Board #787: register :Symbol so a resolve-and-save onto an
            # existing Symbol node overlays (keeps file/line/...) instead of
            # wiping it. None for the live watcher (unchanged). graphiti-core
            # ignores a None entity_types, so this is safe to always pass.
            **({"entity_types": entity_types} if entity_types else {}),
        )
    except Exception:
        logger.warning(
            "write_decision: add_episode failed for %s; NL search surface "
            "skipped, structured Decision node is still written below",
            episode_name,
            exc_info=True,
        )

    # Deterministic, LLM-free structured node — mirrors
    # _merge_structured_symbol's pattern for Symbols.
    #
    # ROOT CAUSE (found live, NOT just the elementId() parse failure): the
    # previous code retroactively SET these fields onto whatever node shared
    # `result.episode.uuid` — but `add_episode()`'s returned `episode` is the
    # :Episodic node it just created, and Entity nodes (what
    # `MATCH (n:Entity) WHERE n.uuid = $uuid` requires) are graphiti's own
    # NL-extracted concepts — a *different* node with a *different* uuid.
    # That WHERE clause can never match a row, on Neo4j or FalkorDB alike,
    # elementId() or not: dropping elementId() alone (the original FIX 2)
    # only fixed the PARSE error and left the SET silently matching zero
    # rows (verified live: no "post-hoc SET failed" warning logged, yet 0
    # nodes end up with type='Decision' after a real 10-commit ingest).
    # Every memex query that reads decisions (mcp_server/queries.py's
    # get_recent_decisions_raw/get_symbol_decisions/
    # count_unvalidated_decisions, mcp_server/tools_write.py's
    # supersede/corroborate paths) requires `MATCH (d:Entity) WHERE
    # d.type = 'Decision' OR d.name CONTAINS 'Decision'` — so a genuinely
    # separate :Entity node, created directly with a uuid we control (never
    # ambiguous, never requires a second lookup), is the only shape that
    # actually satisfies them. Best-effort: a failure here is logged, not
    # raised, matching every other structured-write call site in this file.
    # Council fix 2: deterministic uuid5(repo, commit_sha, text-slug) rather
    # than a fresh uuid4() per call — see _DECISION_MERGE_QUERY's docstring.
    decision_uuid = str(uuid_module.uuid5(
        _DECISION_UUID_NAMESPACE,
        f"{repo_root or ''}|{commit_sha}|{_decision_identity_slug(decision.text)}",
    ))
    try:
        await client.driver.execute_query(
            _DECISION_MERGE_QUERY,
            params={
                "uuid": decision_uuid,
                "name": episode_name,
                "text": decision.text,
                "rationale": decision.rationale,
                "scope": decision.scope,
                "now": now,
                "source": decision_source,
                "source_commit": commit_sha,
                "confidence": confidence,
                "validated": validated,
                "base_confidence": base_confidence,
                # Council fix 1 — the rationale link: MOTIVATES edges to
                # every changed-file Module, written in this same query/
                # transaction. See _DECISION_MERGE_QUERY's docstring.
                "modules": modules or [],
                "repo": repo_root,
            },
        )
    except Exception:
        logger.warning(
            "structured Decision MERGE failed for %s; the decision's NL "
            "episode is still written but type='Decision' metadata and its "
            "MOTIVATES edges are missing until backfilled",
            episode_name,
            exc_info=True,
        )

    # Board #803 (council default): write a SEARCHABLE RELATES_TO twin of the
    # MOTIVATES edges so the rationale is findable (graphiti search only
    # traverses RELATES_TO). OPT-IN (like versioned=/entity_types=): the #787
    # ingest passes searchable_rationale=True; the live watcher leaves it False
    # so its call pattern (and every existing test) is unchanged. No-op if the
    # structured MERGE above failed — the shadow MATCHes the Decision by uuid.
    if searchable_rationale:
        await _write_motivates_shadow(
            client, decision_uuid, decision, modules or [], repo_root, now,
            config.unified_group_id,
        )


# ---------------------------------------------------------------------------
# v0.3.1 Deliverable 5 — IMPORTS edges + Dependency nodes from lockfiles
# ---------------------------------------------------------------------------


#: Initial confidence anchor for IMPORTS edges. Lockfile-derived edges
#: come from deterministic AST parsing — higher than the watcher Decision
#: default (0.6) but not 1.0 so two-regime decay can still surface
#: long-stale imports.
_IMPORT_EDGE_BASE_CONFIDENCE = 0.9

_DEPENDENCY_BASE_CONFIDENCE = 0.95


async def write_lockfile_delta(
    repo_root: str,
    dependencies: list[Dependency],
    imports: list[tuple[str, str, dict]],
) -> dict[str, int]:
    """Persist Dependency nodes + Module IMPORTS edges from lockfile parsing.

    Wired into :func:`memex.watcher.handlers.handle_lockfile_change`. Both
    writes follow the v0.3.0 hybrid pattern (Q1 in ARCHITECTURE §4):

    - Dependency nodes use ``client.add_episode`` so the NL Graphiti
      pipeline can still surface them in search, with a post-hoc Cypher
      SET for the v0.3.0 fields (``write_policy``, ``last_reinforced_at``,
      ``base_confidence``) that Graphiti doesn't parse from NL.

    - IMPORTS edges are pure structure (no NL surface). We MERGE the
      Module endpoints (idempotent — the watcher may have already
      created them) and MERGE the ``IMPORTS`` edge with the v0.3.0
      fields written inline so future composite-reranker filters
      (``WHERE r.expired_at IS NULL``) see them.

    Returns a ``{"deps_written": N, "edges_written": M}`` summary used by
    the watcher log line so re-runs are observable.
    """
    client = await get_graph_client()
    config = get_config()
    now = datetime.now(UTC)
    deps_written = 0
    edges_written = 0

    # 1. Dependencies — episode + post-hoc SET
    for dep in dependencies:
        episode_name = f"dependency_{dep.ecosystem}_{dep.name}"
        try:
            result = await client.add_episode(
                name=episode_name,
                episode_body=(
                    f"Dependency: {dep.name} version {dep.version} "
                    f"({dep.ecosystem} ecosystem)."
                ),
                source_description=f"lockfile scan in {repo_root}",
                reference_time=now,
                group_id=config.unified_group_id,
                # "Dependency: <name> version ..." — same leading-label trap as
                # write_decision above. See that call site for the measurement.
                source=EpisodeType.text,
            )
        except Exception:
            logger.warning(
                "lockfile: add_episode failed for dependency %s",
                episode_name,
                exc_info=True,
            )
            continue

        episode_uuid = getattr(getattr(result, "episode", None), "uuid", None)
        # Same FalkorDB-portable identity match as write_decision's set_query
        # above — `elementId()` is Neo4j-only and fails FalkorDB's Cypher
        # parser outright.
        set_query = """
        MATCH (n:Entity)
        WHERE n.uuid = $uuid
        SET n.type = 'Dependency',
            n.ecosystem = $ecosystem,
            n.version = $version,
            n.last_updated = $now,
            n.last_reinforced_at = $now,
            n.base_confidence = $base_confidence,
            n.write_policy = 'locked',
            n.repo_path = $repo,
            n.access_count = coalesce(n.access_count, 0)
        """
        if episode_uuid is None:
            logger.debug(
                "lockfile: dependency %s missing episode.uuid; "
                "skipping v0.3.0 SET to avoid mis-targeting",
                episode_name,
            )
        else:
            try:
                await client.driver.execute_query(
                    set_query,
                    params={
                        "uuid": episode_uuid,
                        "ecosystem": dep.ecosystem,
                        "version": dep.version,
                        "now": now,
                        "base_confidence": _DEPENDENCY_BASE_CONFIDENCE,
                        "repo": repo_root,
                    },
                )
                deps_written += 1
            except Exception:
                logger.warning(
                    "lockfile: post-hoc SET failed for dependency %s",
                    episode_name,
                    exc_info=True,
                )

    # 2. IMPORTS edges — MERGE Module endpoints + MERGE edge
    # We rely on the watcher having created Entity rows for these modules
    # under name=module_path; if they don't exist yet we create them with
    # type='Module' so the edge always has both endpoints. The watcher's
    # symbol pass will fill in language / created_at on its next visit.
    edge_query = """
    MERGE (src:Entity {name: $from_path, repo_path: $repo})
      ON CREATE SET src.type = 'Module',
                    src.created_at = $now,
                    src.write_policy = 'locked',
                    src.access_count = 0
    MERGE (dst:Entity {name: $to_path, repo_path: $repo})
      ON CREATE SET dst.type = 'Module',
                    dst.created_at = $now,
                    dst.write_policy = 'locked',
                    dst.access_count = 0
    MERGE (src)-[r:IMPORTS]->(dst)
      ON CREATE SET r.created_at = $now,
                    r.base_confidence = $base_confidence,
                    r.kind = $kind,
                    r.expired_at = NULL,
                    r.last_reinforced_at = $now
      ON MATCH SET  r.last_reinforced_at = $now,
                    r.kind = $kind,
                    r.expired_at = NULL
    """
    for from_module, to_module, meta in imports:
        from_path = _dotted_to_repo_path(from_module)
        to_path = _dotted_to_repo_path(to_module)
        if not from_path or not to_path or from_path == to_path:
            continue
        try:
            await client.driver.execute_query(
                edge_query,
                params={
                    "from_path": from_path,
                    "to_path": to_path,
                    "repo": repo_root,
                    "now": now,
                    "base_confidence": _IMPORT_EDGE_BASE_CONFIDENCE,
                    "kind": meta.get("kind", "import"),
                },
            )
            edges_written += 1
        except Exception:
            logger.warning(
                "lockfile: IMPORTS edge write failed for %s -> %s",
                from_path,
                to_path,
                exc_info=True,
            )

    return {"deps_written": deps_written, "edges_written": edges_written}


def _dotted_to_repo_path(name: str) -> str:
    """Map a dotted Python module name back to its repo-relative path.

    ``extract_module_imports`` returns dotted names (``memex.watcher.handlers``);
    Module nodes elsewhere are stored under ``name=<repo-relative path>``
    (``memex/watcher/handlers.py``). We append ``.py`` because the import
    extractor only walks Python today. If the input already looks like a
    path (contains ``/`` or ends with a known extension), passthrough.
    """
    if not name:
        return ""
    if "/" in name or "\\" in name:
        return str(name).replace("\\", "/")
    if "." in name and not name.endswith(".py"):
        return name.replace(".", "/") + ".py"
    return name
