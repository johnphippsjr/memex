"""Board #788 GAP 1 — the k8s/infra YAML structural extractor.

DB-free tests over memex.extractor.k8s. The end-to-end proof (real homek8-shaped
manifests -> resource nodes with :Entity:Symbol + card embedding, plus resolved
REFERENCES edges) was run live against memex-fork-test on 2026-08-02.
"""

from memex.extractor.k8s import extract_k8s_symbols, ResourceRef

DEPLOY = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: falkordb
  namespace: mcp
spec:
  template:
    spec:
      serviceAccountName: falkordb-sa
      containers:
      - name: falkordb
        image: falkordb/falkordb:latest
        envFrom:
        - secretRef:
            name: litellm-master-key
        - configMapRef:
            name: falkordb-config
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: data-falkordb-0
"""


def test_a_k8s_doc_becomes_a_resource_symbol():
    ex = extract_k8s_symbols("cluster/mcp/B5-falkordb.yaml", DEPLOY)
    assert len(ex.symbols) == 1
    sym = next(iter(ex.symbols.values()))
    assert sym.name == "Deployment/mcp/falkordb"   # Kind/namespace/name identity
    assert sym.kind == "resource"
    assert "apiVersion=apps/v1" in sym.signature
    assert sym.file == "cluster/mcp/B5-falkordb.yaml"


def test_references_are_extracted_with_their_via_field():
    ex = extract_k8s_symbols("d.yaml", DEPLOY)
    refs = next(iter(ex.refs.values()))
    got = {(r.kind, r.name, r.via) for r in refs}
    assert ("Secret", "litellm-master-key", "secretRef") in got
    assert ("ConfigMap", "falkordb-config", "configMapRef") in got
    assert ("PersistentVolumeClaim", "data-falkordb-0", "claimName") in got
    assert ("ServiceAccount", "falkordb-sa", "serviceAccountName") in got
    # the container image is captured too, but as an external Image ref
    assert ("Image", "falkordb/falkordb:latest", "image") in got


def test_cluster_scoped_resource_omits_namespace_in_identity():
    doc = "apiVersion: v1\nkind: Namespace\nmetadata:\n  name: mcp\n"
    ex = extract_k8s_symbols("ns.yaml", doc)
    sym = next(iter(ex.symbols.values()))
    assert sym.name == "Namespace/mcp"   # no namespace segment


def test_multi_document_yaml_yields_one_symbol_per_resource():
    multi = DEPLOY + "---\n" + "apiVersion: v1\nkind: Service\nmetadata:\n  name: falkordb\n  namespace: mcp\n"
    ex = extract_k8s_symbols("both.yaml", multi)
    names = {s.name for s in ex.symbols.values()}
    assert names == {"Deployment/mcp/falkordb", "Service/mcp/falkordb"}


def test_non_k8s_yaml_yields_nothing_not_a_wrong_guess():
    """Honesty contract (#783): a Helm values file / CI config / plain data map
    is NOT a k8s resource and must produce ZERO symbols, not a bogus one."""
    for junk in (
        "foo: bar\nbaz:\n  - 1\n  - 2\n",              # plain data
        "replicaCount: 3\nimage:\n  tag: v1\n",          # helm values (no kind)
        "name: build\non: [push]\njobs: {}\n",           # CI config
    ):
        ex = extract_k8s_symbols("x.yaml", junk)
        assert ex.symbols == {}, f"non-k8s YAML produced symbols: {junk!r}"


def test_a_kind_without_a_name_is_not_a_resource():
    """A List/kustomization fragment (kind+apiVersion but no metadata.name) is
    not an addressable resource."""
    doc = "apiVersion: v1\nkind: List\nitems: []\n"
    assert extract_k8s_symbols("l.yaml", doc).symbols == {}


def test_malformed_yaml_does_not_raise():
    """Same 'zero symbols rather than a crash' contract as the code path."""
    ex = extract_k8s_symbols("bad.yaml", "apiVersion: v1\nkind: Pod\n  bad: : indent\n")
    assert ex.symbols == {}


def test_resource_kind_is_valid_on_the_symbol_schema():
    """The extractor emits kind='resource'; the Symbol schema must accept it, or
    every infra node raises MemexSchemaError at write time."""
    from memex.graph.schema import Symbol
    s = Symbol(name="Deployment/x", kind="resource", signature="k=Deployment", file="d.yaml", line=1)
    assert s.kind == "resource"
