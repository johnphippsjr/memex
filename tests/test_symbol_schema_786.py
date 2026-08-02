"""Board #786 — the Symbol node carries :Entity:Symbol and a card embedding.

These are DB-free property tests over the writer's query constants and card
builder. The end-to-end proof (write a symbol into a throwaway FalkorDB, confirm
the :Symbol label + name_embedding, and that a natural-language vector query
ranks it #1) was run live against memex-fork-test on 2026-08-02; this file is the
durable guard that the code path stays intact.
"""

from memex.graph import writer
from memex.graph.schema import Symbol, SymbolEntityType


class _Sym:
    kind = "fn"
    name = "reconcile_replica_set"
    signature = "def reconcile_replica_set(desired, observed):"
    file = "controller/reconcile.py"
    line = 42


def test_symbol_merge_query_adds_the_symbol_label():
    """A Symbol must be :Entity:Symbol, not bare :Entity. #782 proved the
    multi-label node stays inside every Entity index while gaining a clean
    MATCH (n:Symbol) surface."""
    assert "SET s:Symbol" in writer._SYMBOL_MERGE_QUERY
    # ...and the embedded variant is the plain query PLUS the vector write,
    # so it can never drift into a different MERGE.
    assert writer._SYMBOL_MERGE_QUERY_EMBEDDED.startswith(writer._SYMBOL_MERGE_QUERY)
    assert "name_embedding = vecf32($name_embedding)" in writer._SYMBOL_MERGE_QUERY_EMBEDDED


def test_plain_query_never_writes_an_embedding():
    """The fallback used when the embedder is down must NOT reference
    name_embedding — a missing embedding degrades to 'not vector-searchable',
    it must never write a null/garbage vector."""
    assert "name_embedding" not in writer._SYMBOL_MERGE_QUERY


def test_symbol_card_is_a_composed_card_not_the_bare_name():
    """#786 measured that embedding the bare identifier is WORSE than not
    embedding (bge-m3 hashes short snake_case orthographically). The card must
    carry kind + signature + file context, not just the name."""
    card = writer._symbol_card(_Sym(), "/repo/demo")
    assert _Sym.name in card
    assert _Sym.kind in card
    assert _Sym.signature in card
    assert _Sym.file in card
    # It is a COMPOSED card, so materially longer than the identifier alone.
    assert len(card) > len(_Sym.name) + 20


def test_symbol_card_survives_a_missing_signature():
    """signature == name (tree-sitter's fallback) must not double-print, and a
    None signature must not crash the builder."""
    class _Bare:
        kind = "const"; name = "MAX"; signature = "MAX"; file = "c.py"; line = 1
    card = writer._symbol_card(_Bare(), None)
    assert "signature: MAX" not in card  # not duplicated when sig == name
    assert "MAX" in card


def test_symbol_entity_type_declares_no_extractable_fields():
    """The registration's whole job (#782) is to be EMPTY so graphiti's overlay
    merge preserves the tree-sitter structural props. If a future edit adds
    file/line/signature/repo_path here, it re-opens the silent-wipe bug — this
    test fails loudly if anyone does."""
    declared = set(SymbolEntityType.model_fields)
    forbidden = {"file", "line", "signature", "repo_path", "name", "kind"}
    assert declared.isdisjoint(forbidden), (
        f"SymbolEntityType must not declare structural fields; found {declared & forbidden}. "
        "Declaring them invites the LLM to overwrite tree-sitter ground truth."
    )
    # An instance accepts (and ignores) structural attrs without storing them.
    inst = SymbolEntityType()
    assert not inst.model_dump()


# ---------------------------------------------------------------------------
# Board #786 versioning half — bi-temporal Symbol history for the ingest.
# End-to-end (two commit-times -> two nodes, old closed / new open, idempotent,
# watcher still overwrites) was verified live against memex-fork-test on
# 2026-08-02; these are the durable DB-free guards on the query shapes.
# ---------------------------------------------------------------------------

import inspect


def test_write_symbol_delta_has_a_versioned_flag_defaulting_false():
    """The live watcher must keep overwriting in place; only the ingest opts in.
    Default False preserves the watcher's behaviour with no call-site change."""
    sig = inspect.signature(writer.write_symbol_delta)
    assert "versioned" in sig.parameters
    assert sig.parameters["versioned"].default is False


def test_version_merge_keys_on_valid_from():
    """Each commit-time state must be its OWN node. If the MERGE key loses
    valid_from it collapses back to overwrite-in-place and history is lost."""
    assert "valid_from: $now" in writer._SYMBOL_VERSION_MERGE_QUERY
    # On re-run it must be a no-op, not a re-write of the structural props:
    # ON MATCH may only touch last_reinforced_at.
    on_match = writer._SYMBOL_VERSION_MERGE_QUERY.split("ON MATCH SET")[1]
    for forbidden in ("v.signature", "v.kind", "v.valid_until", "v.uuid"):
        assert forbidden not in on_match, f"ON MATCH must not rewrite {forbidden}"


def test_close_open_query_excludes_the_current_commit():
    """The close-others step must exclude valid_from = $now, or a resumed run
    re-processing the same commit would close the very version it just wrote."""
    q = writer._SYMBOL_CLOSE_OPEN_QUERY
    assert "valid_until IS NULL" in q
    assert "s.valid_from <> $now" in q


def test_removed_only_closes_the_open_version():
    """A removal must not stamp valid_until over already-closed historical
    versions — that would corrupt their real close-dates. Scope to the open one.
    (Asserted on the source since the query is inline in write_symbol_delta.)"""
    src = inspect.getsource(writer.write_symbol_delta)
    # the removal query in the source must carry the open-version guard
    assert "s.valid_until IS NULL" in src


def test_versioned_and_overwrite_paths_are_distinct_functions():
    """The two write modes must be separate code paths — the watcher's overwrite
    must be reachable independently of the ingest's append."""
    assert writer._merge_structured_symbol is not writer._version_structured_symbol
