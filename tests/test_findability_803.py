"""Board #803 (council default) — rationale findability + UNRESOLVED_REF.

DB-free tests over the writer's #803 additions:
  * MOTIVATES gets a deterministic SEARCHABLE RELATES_TO shadow (graphiti search
    only traverses RELATES_TO), leading with the rationale as the `fact`.
  * pure-structure resource refs that don't resolve to exactly one in-repo
    target are RECORDED as UNRESOLVED_REF (traversal-only) instead of dropped.
The live proof (the shadow edge is written in graphiti's exact RELATES_TO shape
with fact+fact_embedding+group_id, and an UnresolvedRef node/edge appears) ran
against memex-fork-test on a throwaway graph — see the board note.
"""

import pytest

from memex.graph import writer as w


def test_module_uuid_is_deterministic_and_path_scoped():
    a = w._module_uuid("/repoA", "app/main.py")
    assert a == w._module_uuid("/repoA", "app/main.py")           # stable across calls
    assert a != w._module_uuid("/repoB", "app/main.py")           # repo-scoped
    assert a != w._module_uuid("/repoA", "app/other.py")          # path-scoped


def test_motivates_shadow_is_a_searchable_relates_to():
    q = w._MOTIVATES_SHADOW_QUERY
    assert "RELATES_TO" in q
    assert "r.name = 'MOTIVATES'" in q
    assert "r.fact = $fact" in q
    assert "r.group_id = $group_id" in q          # else post-filtered out of search
    assert "r.source_node_uuid" in q and "r.target_node_uuid" in q
    assert "vecf32($fact_embedding)" in w._MOTIVATES_SHADOW_QUERY_EMBEDDED


def test_unresolved_ref_is_recorded_traversal_only():
    q = w._UNRESOLVED_REF_QUERY
    assert "type: 'UnresolvedRef'" in q
    assert "r.unresolved = true" in q
    assert "REFERENCES" in q
    # traversal-only: the marker must NOT carry a group_id or embedding, so it
    # stays OUT of semantic search (council: pure structure stays traversal-only)
    assert "group_id" not in q
    assert "fact_embedding" not in q


@pytest.mark.asyncio
async def test_write_motivates_shadow_leads_with_rationale_and_embeds():
    captured = []

    class FakeEmbedder:
        async def create(self, text):
            return [0.0] * 1024

    class FakeDriver:
        async def execute_query(self, q, params=None):
            captured.append((q, params))
            class R:
                records = []
            return R()

    class FakeClient:
        embedder = FakeEmbedder()
        driver = FakeDriver()

    class D:
        text = "Pin redis to <8"
        rationale = "8.1 adds a kwarg FalkorDB's client rejects"

    from datetime import datetime, timezone
    await w._write_motivates_shadow(
        FakeClient(), "dec-uuid-1", D(), ["app/main.py"], "/r",
        datetime(2021, 1, 1, tzinfo=timezone.utc), "grp",
    )
    assert captured, "shadow wrote no query"
    q, params = captured[-1]
    assert "RELATES_TO" in q and "vecf32" in q          # embedder succeeded -> embedded form
    assert params["fact"].startswith("Pin redis to <8")  # rationale-led, not a node label
    assert "app/main.py" in params["fact"]
    assert params["group_id"] == "grp"
    assert params["decision_uuid"] == "dec-uuid-1"


@pytest.mark.asyncio
async def test_resource_ref_zero_resolution_records_unresolved(monkeypatch):
    from memex.extractor.k8s import extract_k8s_symbols

    calls = []

    class _Rec:
        def get(self, k, d=None):
            return 0   # REFERENCES resolution -> n=0 (nothing resolved)

    class FakeDriver:
        async def execute_query(self, q, params=None):
            calls.append((q, params))
            class R:
                records = [_Rec()]
            return R()

    class FakeClient:
        driver = FakeDriver()

    async def _fake_gc():
        return FakeClient()

    monkeypatch.setattr(w, "get_graph_client", _fake_gc)

    doc = (
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web\n  namespace: app\n"
        "spec:\n  template:\n    spec:\n      containers:\n      - name: c\n        image: x:1\n"
        "        envFrom:\n        - secretRef:\n            name: external-secret\n"
    )
    extract = extract_k8s_symbols("d.yaml", doc)
    await w.write_resource_ref_edges(extract, repo_root="/r")

    # the Secret ref resolves to nothing -> an UNRESOLVED_REF query must fire
    assert any("UnresolvedRef" in q for q, _ in calls), \
        "unresolved Secret ref was dropped silently instead of recorded"
