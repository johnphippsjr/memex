"""Board #801 (deeper rebuild) — stable symbol sid + edge-fact lifetimes.

DB-free tests over the sid minting, the intra-file rename matcher, and the
bitemporal query shapes. The live proof (a repo that renames a function AND
moves a file collapses to ONE connected sid lineage with RENAMED_TO edges and
DEFINED_IN intervals, not orphaned nodes) ran against memex-fork-test on a
throwaway graph — see the board note.
"""

import pytest

from memex.graph import writer as w
from memex.graph.writer import _mint_sid
from memex.graph.schema import Symbol
from memex.extractor.treesitter import SymbolDelta
from memex.ingest_history import _detect_intrafile_renames


def test_mint_sid_deterministic_and_scoped():
    a = _mint_sid("/r", "c1", "foo", "a.py")
    assert a == _mint_sid("/r", "c1", "foo", "a.py")     # stable
    assert a != _mint_sid("/r", "c1", "foo", "b.py")     # file-scoped at first sighting
    assert a != _mint_sid("/r", "c2", "foo", "a.py")     # commit-scoped
    assert a != _mint_sid("/other", "c1", "foo", "a.py") # repo-scoped


def test_bitemporal_queries_carry_the_four_lifetime_edges():
    assert "MERGE (s:Entity {sid: $sid})" in w._BITEMPORAL_NODE_QUERY
    assert "coalesce(ip.sid, rn.sid)" in w._RESOLVE_SID_QUERY   # in-place > rename-source
    assert "INTRODUCED_IN" in w._INTRODUCED_IN_QUERY
    assert "DEFINED_IN" in w._OPEN_DEFINED_IN_INTERVAL_QUERY
    assert "REMOVED_IN" in w._REMOVED_IN_QUERY
    assert "RENAMED_TO" in w._RENAMED_TO_QUERY
    assert "vecf32($name_embedding)" in w._BITEMPORAL_NODE_QUERY_EMBEDDED


def _sym(name, sig, file="a.py"):
    return Symbol(name=name, kind="fn", signature=sig, file=file, line=1)


def test_intrafile_rename_matches_similar_and_drops_the_removal():
    delta = SymbolDelta(added=[_sym("bar", "def bar(x):")], removed=[_sym("foo", "def foo(x):")])
    renames = _detect_intrafile_renames(delta)
    assert ("bar", "a.py") in renames
    old_name, old_file, ratio = renames[("bar", "a.py")]
    assert old_name == "foo" and old_file == "a.py"
    assert ratio >= 0.6
    # the rename must NOT also be recorded as a delete
    assert delta.removed == []


def test_intrafile_rename_ignores_dissimilar():
    delta = SymbolDelta(
        added=[_sym("handler", "async def handler(request, context, options):")],
        removed=[_sym("x", "x = 1")],
    )
    renames = _detect_intrafile_renames(delta)
    assert renames == {}
    assert len(delta.removed) == 1   # kept as a genuine delete


def test_intrafile_rename_respects_file_boundary():
    delta = SymbolDelta(
        added=[_sym("bar", "def bar():", file="new.py")],
        removed=[_sym("foo", "def foo():", file="old.py")],
    )
    renames = _detect_intrafile_renames(delta)
    assert renames == {}   # different files -> not an in-place rename (that's a move)
    assert len(delta.removed) == 1
