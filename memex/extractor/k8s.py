"""Structural extractor for Kubernetes / infra YAML (board #788 GAP 1).

The council's Static-Analysis seat: infra structure is a DIFFERENT SHAPE from
code and needs its own extractor, not the tree-sitter code path. A k8s repo like
homek8 has no functions or classes — it has RESOURCES with
``apiVersion``/``kind``/``namespace``/``name`` identity and explicit references
(``image``, ``secretRef``/``configMapRef``, ``claimName``, service ``selector``).
That is a real, DETERMINISTIC dependency graph — no LLM needed.

This produces the same :class:`~memex.graph.schema.Symbol` objects the code path
does, so infra resources flow through the identical writer, gain the
:Entity:Symbol label + card embedding + bi-temporal versioning built for #786,
and become searchable exactly like code symbols. The resource identity is the
Symbol ``name`` (``Kind/namespace/name``); ``kind`` is the new ``"resource"``
value; the reference targets are returned separately so the caller can build
edges.

Scope, stated honestly:
  * Kubernetes manifests only (a doc with both ``apiVersion`` and ``kind``).
    Non-k8s YAML — CI configs, Helm values, plain data — yields NOTHING rather
    than guessing, exactly the honesty #783 restored for unmapped extensions.
  * HCL/Terraform is NOT handled here (needs a different parser); the task title
    lists it but the operator's target is homek8, which is k8s YAML.
  * References are extracted by walking well-known k8s fields, not by
    understanding every CRD. Unknown CRDs still get a resource node (identity is
    generic); only their references may be missed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import yaml

from memex.graph.schema import Symbol

logger = logging.getLogger(__name__)


@dataclass
class ResourceRef:
    """One deterministic reference from a resource to another named thing.

    ``kind`` is the referent's k8s kind when known (``Secret``, ``ConfigMap``,
    ``PersistentVolumeClaim``, ``Image``…), ``name`` its name/value. ``via`` is
    the field that produced it (``secretRef``, ``image``, ``claimName``…), kept
    so the caller can label the edge and so this stays auditable.
    """

    kind: str
    name: str
    via: str


@dataclass
class K8sExtract:
    symbols: dict[str, Symbol] = field(default_factory=dict)
    #: symbol-key -> list of ResourceRef it points at
    refs: dict[str, list[ResourceRef]] = field(default_factory=dict)


def _resource_identity(doc: dict[str, Any]) -> tuple[str, str] | None:
    """Return (identity_name, signature) for a k8s doc, or None if it is not a
    k8s resource. Identity is ``Kind/namespace/name`` (namespace omitted when the
    resource is cluster-scoped / has none) — stable across re-ingest and unique
    per resource within a repo."""
    kind = doc.get("kind")
    api = doc.get("apiVersion")
    if not isinstance(kind, str) or not isinstance(api, str):
        return None
    meta = doc.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    name = meta.get("name")
    if not isinstance(name, str) or not name:
        # A kind+apiVersion with no metadata.name is a List, a kustomization
        # fragment, or malformed — not an addressable resource.
        return None
    ns = meta.get("namespace")
    identity = f"{kind}/{ns}/{name}" if isinstance(ns, str) and ns else f"{kind}/{name}"
    sig = f"apiVersion={api} kind={kind}" + (
        f" namespace={ns}" if isinstance(ns, str) and ns else ""
    ) + f" name={name}"
    return identity, sig


def _walk_refs(node: Any, out: list[ResourceRef]) -> None:
    """Depth-first walk collecting the well-known k8s cross-references. Field
    names are matched by the k8s convention, so this works for Deployments,
    StatefulSets, CronJobs, Pods, etc. without enumerating every workload kind."""
    if isinstance(node, dict):
        for key, val in node.items():
            # container image (may be a bare string or under `image:`)
            if key == "image" and isinstance(val, str) and val:
                out.append(ResourceRef("Image", val, "image"))
            # *.secretName -> a Secret (volumes, TLS, imagePullSecrets shapes)
            elif key == "secretName" and isinstance(val, str):
                out.append(ResourceRef("Secret", val, key))
            elif key in ("secretRef", "secretKeyRef") and isinstance(val, dict):
                nm = val.get("name")
                if isinstance(nm, str):
                    out.append(ResourceRef("Secret", nm, key))
            # configMapRef / configMapKeyRef / configMap volume -> a ConfigMap
            elif key in ("configMapRef", "configMapKeyRef") and isinstance(val, dict):
                nm = val.get("name")
                if isinstance(nm, str):
                    out.append(ResourceRef("ConfigMap", nm, key))
            elif key == "configMap" and isinstance(val, dict):
                nm = val.get("name")
                if isinstance(nm, str):
                    out.append(ResourceRef("ConfigMap", nm, "configMap"))
            # PVC claim
            elif key == "claimName" and isinstance(val, str):
                out.append(ResourceRef("PersistentVolumeClaim", val, "claimName"))
            elif key == "serviceAccountName" and isinstance(val, str):
                out.append(ResourceRef("ServiceAccount", val, "serviceAccountName"))
            else:
                _walk_refs(val, out)
    elif isinstance(node, list):
        for item in node:
            _walk_refs(item, out)


def extract_k8s_symbols(file_path: str, content: str) -> K8sExtract:
    """Parse a (possibly multi-document) k8s YAML file into resource Symbols.

    Never raises on bad YAML: a parse failure is logged and yields an empty
    result, matching the code path's "zero symbols rather than a crash" contract.
    """
    result = K8sExtract()
    if not content or not content.strip():
        return result

    try:
        docs = list(yaml.safe_load_all(content))
    except yaml.YAMLError:
        logger.warning(
            "k8s extractor: YAML parse failed for %s; zero resources for this "
            "file rather than raising", file_path, exc_info=True,
        )
        return result

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        ident = _resource_identity(doc)
        if ident is None:
            continue
        identity, signature = ident
        sym = Symbol(
            name=identity,
            kind="resource",
            signature=signature,
            file=file_path,
            line=1,  # YAML docs have no meaningful single defining line
        )
        key = f"{identity}:resource"
        result.symbols[key] = sym

        refs: list[ResourceRef] = []
        _walk_refs(doc, refs)
        # de-dup (name,kind,via) — the same image/secret often appears twice
        seen = set()
        deduped = []
        for r in refs:
            sig = (r.kind, r.name, r.via)
            if sig not in seen:
                seen.add(sig)
                deduped.append(r)
        if deduped:
            result.refs[key] = deduped

    return result
