import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import tree_sitter as ts
import tree_sitter_language_pack as tslp
from memex.graph.schema import Symbol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BOARD #1038 - tags-query symbol extraction (replaces the tslp process() path
# for JS/TS/TSX/JSX, which could not see a function's name at all).
#
# WHY QUERIES AND NOT tslp.process(): process() does not carry the BINDING
# identifier. `const foo = () => {}` came back with name=None, Symbol.name is a
# required str, pydantic raised, and the whole file's symbols were discarded.
# tree-sitter's OFFICIAL answer to this is the tags.scm convention, whose
# JavaScript query contains our exact case:
#   (variable_declarator value: [(arrow_function)(function_expression)]) @definition.function
#
# WHY VENDORED (queries/*.scm) AND NOT `import tree_sitter_javascript`:
# depending on the wheels would put THREE independently-versioned grammar
# sources in play (tslp's compiled grammar + two query wheels). We compile the
# vendored text against tslp's grammars, so there is exactly one grammar source.
# Bumping a vendored file means RE-MEASURING the fixture counts.
#
# WHY .ts/.tsx GET THREE FILES CONCATENATED: upstream's TypeScript tags.scm is
# 573 bytes and is a SUPPLEMENT to the JavaScript one (TS inherits the JS
# grammar). MEASURED: used alone on a React/TSX sample it returns ZERO
# definitions. JS+TS+supplement returns 11 on the same sample.
#
# WHY COMPILED AT IMPORT: tree_sitter 0.26.0 has a confirmed bug where ONE
# failed query compile poisons that query text for the life of the process,
# with a misleading error, transitively across languages. So every query is
# compiled once, here, and an error is raised at import rather than being
# discovered mid-ingest.
# ---------------------------------------------------------------------------
_QUERY_DIR = os.path.join(os.path.dirname(__file__), "queries")


def _read_query(name: str) -> str:
    with open(os.path.join(_QUERY_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def _build_tag_queries() -> Dict[str, "ts.Query"]:
    js = _read_query("javascript-tags.scm")
    tsq = _read_query("typescript-tags.scm")
    sup = _read_query("memex-supplement.scm")
    combined_ts = "\n".join((js, tsq, sup))
    sources = {
        "javascript": js,
        "typescript": combined_ts,
        "tsx": combined_ts,
    }
    built: Dict[str, ts.Query] = {}
    for lang, src in sources.items():
        # Deliberately NOT wrapped in try/except: a broken query must stop the
        # process at import, not silently degrade to zero symbols at runtime.
        built[lang] = ts.Query(tslp.get_language(lang), src)
    return built


_TAG_QUERIES: Dict[str, "ts.Query"] = _build_tag_queries()

# @definition.<suffix> -> the Symbol.kind bucket. FROZEN to today's two buckets
# on purpose (operator/data-integrity): introducing richer kinds now would change
# every symbol key and present as a repo-wide fake refactor in the history. A
# suffix that is not in this map raises rather than being silently dropped.
_ROLE_TO_KIND = {
    "function": "fn",
    "method": "fn",
    "class": "class",
    "interface": "class",
    "type": "class",
    "enum": "class",
    "module": None,   # skipped: not a code symbol in today's model
}

@dataclass
class SymbolDelta:
    added: List[Symbol] = field(default_factory=list)
    removed: List[Symbol] = field(default_factory=list)
    modified: List[Symbol] = field(default_factory=list)


@dataclass
class CallEdge:
    """A resolved call-site: function `caller` (defined in `file`) calls the
    name `callee` at 1-indexed `line`. `callee` is a bare name — resolution to
    a concrete target Symbol node happens at write time (writer.write_call_edges).
    """
    caller: str
    callee: str
    file: str
    line: int


# tree-sitter call-expression queries, per language. Captures the *callee*
# name (the simple identifier, or the trailing attribute of `a.b.c()`).
# v0.3.7 Layer 2 ships Python only — memex's own codebase is Python and the
# grammars differ per language; others fall through to [] (no edges, no error).
_CALL_QUERIES: Dict[str, str] = {
    "python": "(call function: [(identifier) @callee "
              "(attribute attribute: (identifier) @callee)])",
}


def _flatten_functions(items, acc: List[Tuple[str, int, int]]) -> None:
    """Collect (name, start_line, end_line) for every function/method symbol,
    recursing into class bodies. Lines are 0-indexed (tree-sitter convention)."""
    for it in items:
        kind = str(it.kind).lower()
        if "function" in kind or "method" in kind:
            acc.append((it.name, it.span.start_line, it.span.end_line))
        if it.children:
            _flatten_functions(it.children, acc)


def extract_calls(file_path: str, content: str, language: str = "python") -> List[CallEdge]:
    """Extract intra-file call-sites and map each to its enclosing function.

    Returns one :class:`CallEdge` per (caller, callee) call-site. Calls made at
    module scope (no enclosing function) are skipped — we don't fabricate a
    caller. Unsupported languages return ``[]``.
    """
    query_src = _CALL_QUERIES.get(language)
    if not query_src or not content:
        return []

    try:
        import tree_sitter as ts
        lang = tslp.get_language(language)
        parser = ts.Parser(lang)
        tree = parser.parse(content.encode("utf-8", errors="ignore"))
        query = ts.Query(lang, query_src)
        cursor = ts.QueryCursor(query)
        captures = cursor.captures(tree.root_node)
    except Exception:
        logger.debug("call extraction failed for %s", file_path, exc_info=True)
        return []

    # Enclosing-function spans (0-indexed) for caller resolution.
    try:
        result = tslp.process(content, config=tslp.ProcessConfig(language=language))
        functions: List[Tuple[str, int, int]] = []
        _flatten_functions(result.structure, functions)
    except Exception:
        return []

    raw = content.encode("utf-8", errors="ignore")
    edges: List[CallEdge] = []
    seen: set[Tuple[str, str, int]] = set()

    for node in captures.get("callee", []):
        callee = raw[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        line0 = node.start_point[0]
        # Innermost enclosing function: smallest span that contains the call.
        caller = None
        best = None
        for name, s, e in functions:
            if s <= line0 <= e:
                span = e - s
                if best is None or span < best:
                    best = span
                    caller = name
        if caller is None:
            continue  # module-level call — no caller symbol
        key = (caller, callee, line0)
        if key in seen:
            continue
        seen.add(key)
        edges.append(CallEdge(caller=caller, callee=callee, file=file_path, line=line0 + 1))

    return edges

def _flatten_structure(items) -> list:
    """Recurse into every item's children (class bodies, nested scopes) so a
    class's methods are visited as candidate symbols in their own right, not
    just the top-level def/class statements.

    Council fix 5a: this mirrors `_flatten_functions` above, which already
    recurses for the CALLS graph — `get_symbols_from_content` previously
    iterated `result.structure` flat, so a class's methods were valid CALLS
    *callers* (via _flatten_functions) but never became Symbol nodes of
    their own (via this function): 31.9% of function-like symbols in memex
    itself, 32.7% corpus-wide, confirmed three ways.
    """
    acc = []
    for it in items:
        acc.append(it)
        if it.children:
            acc.extend(_flatten_structure(it.children))
    return acc


def _symbols_from_tags_query(content: str, file_path: str, language_name: str) -> Dict[str, Symbol]:
    """Extract symbols using the vendored tree-sitter tags queries (board #1038).

    Uses QueryCursor.matches(), NOT .captures(). This is load-bearing: .captures()
    returns a FLAT capture list in which @name from a @reference.call pattern is
    indistinguishable from @name on a @definition.function. Measured on the real
    repo, that flattening yields 14,625 reference captures that would have been
    written as fake definitions. .matches() keeps each pattern's captures together,
    so we can require that the match actually carries a @definition.* role.
    """
    symbols: Dict[str, Symbol] = {}
    query = _TAG_QUERIES[language_name]
    lang = tslp.get_language(language_name)
    raw = content.encode("utf-8", errors="ignore")

    parser = ts.Parser(lang)
    tree = parser.parse(raw)
    if tree.root_node.has_error:
        # Honest signal, not a failure: a syntax-error file still yields whatever
        # parsed cleanly, but the caller should not treat the result as complete.
        logger.info("%s: parse reported errors; symbol set may be partial", file_path)

    lines = content.splitlines()
    skipped_unnameable = 0

    for _pattern_index, caps in ts.QueryCursor(query).matches(tree.root_node):
        role = next((c for c in caps if c.startswith("definition.")), None)
        if role is None:
            continue  # a @reference.* match - a call site, not a definition
        suffix = role.split(".", 1)[1]
        if suffix not in _ROLE_TO_KIND:
            raise ValueError(
                "unmapped tags-query role %r in %s - add it to _ROLE_TO_KIND "
                "deliberately rather than letting symbols vanish" % (role, file_path)
            )
        kind = _ROLE_TO_KIND[suffix]
        if kind is None:
            continue  # deliberately not a code symbol in today's model

        name_nodes = caps.get("name")
        if not name_nodes:
            skipped_unnameable += 1
            continue
        node = name_nodes[0]
        name = raw[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")
        if not name:
            skipped_unnameable += 1
            continue

        line_idx = node.start_point[0]
        signature = lines[line_idx].strip() if line_idx < len(lines) else name
        try:
            symbols[f"{name}:{kind}"] = Symbol(
                name=name, kind=kind, signature=signature,
                file=file_path, line=line_idx + 1,
            )
        except Exception:  # noqa: BLE001 - one bad symbol never costs the file
            logger.warning("symbol construction failed for %r in %s", name, file_path,
                           exc_info=True)
            skipped_unnameable += 1

    if skipped_unnameable:
        logger.warning("%s: skipped %d un-nameable definition(s), kept %d",
                       file_path, skipped_unnameable, len(symbols))
    return symbols


def get_symbols_from_content(content: str, file_path: str, language_name: str) -> Dict[str, Symbol]:
    """
    Parses content and extracts symbols using tree-sitter-language-pack high-level API.
    Returns a mapping of symbol 'key' (name:kind) to Symbol object.
    """
    symbols = {}
    if not content:
        return symbols

    # Board #1038: JS/TS/TSX go through the tags queries, which can resolve a
    # function's name from its BINDING. The tslp process() path below cannot,
    # and returned name=None for every anonymous construct.
    if language_name in _TAG_QUERIES:
        return _symbols_from_tags_query(content, file_path, language_name)

    try:
        config = tslp.ProcessConfig(language=language_name)
        result = tslp.process(content, config=config)
    except Exception:
        # Council fix 5b: this used to fail silently — a genuinely
        # unsupported/misconfigured language returned zero symbols and the
        # caller (extract_symbol_delta -> handle_file_change) logged a
        # normal, successful index pass. Log loudly so a repo that silently
        # produces zero symbols on every file is visible, not just quiet.
        logger.warning(
            "tree-sitter parse failed for %s (language=%s); returning zero "
            "symbols for this file rather than raising",
            file_path, language_name, exc_info=True,
        )
        return symbols

    # Board #1038: symbols we could not name, kept so the loss is countable rather than silent.
    skipped_unnameable: list = []

    for item in _flatten_structure(result.structure):
        # Map tree-sitter kinds to our simple kinds
        kind_str = str(item.kind).lower()
        if "function" in kind_str or "method" in kind_str:
            kind = "fn"
        elif "class" in kind_str or "struct" in kind_str or "interface" in kind_str:
            kind = "class"
        else:
            # For Phase 1, we focus on fn and class. 
            # Constants might be 'other' or specific kinds depending on language.
            continue

        # Extract signature: for now, just the line where it starts
        lines = content.splitlines()
        line_idx = item.span.start_line
        signature = lines[line_idx].strip() if line_idx < len(lines) else item.name

        # BOARD #1038 - PER-ITEM ISOLATION. Before this, ONE un-nameable symbol destroyed the
        # WHOLE FILE. item.name is None for anonymous JS/TS constructs (arrow functions bound to
        # a const, `export default function () {}`, `export default class {}`); Symbol.name is a
        # required str, so pydantic raised here, the exception escaped this function AND
        # extract_symbol_delta(), and ingest_history.py caught it, logged a WARNING, and threw
        # away every symbol in the file.
        #
        # MEASURED COST of that design: smokesignals-web ingested 4,773 files and wrote 142
        # symbols with 0 call edges across 734 discarded files - and still exited "Complete".
        #
        # A nameless symbol must now cost AT MOST ITSELF. Note this is deliberately NOT the fix
        # for anonymous naming: the council rejected silent-skip-as-the-answer, because ~90% of a
        # React codebase's functions are nameless arrows bound to consts and skipping them would
        # ship near-total loss reported as success. The real fix is the upstream tree-sitter
        # tags.scm queries, which resolve `const foo = () => {}` to "foo" via
        #   (variable_declarator value: [(arrow_function) (function_expression)]) @definition.function
        # This block is the SAFETY NET underneath that, and it protects every language - including
        # ones nobody has measured yet. Keep it even after tags.scm lands.
        #
        # skipped_unnameable is returned to the caller so this can never be silent again: a file
        # that drops symbols must be able to mark its commit unclean rather than look successful.
        if item.name is None:
            skipped_unnameable.append((file_path, item.span.start_line + 1, kind))
            continue

        try:
            s = Symbol(
                name=item.name,
                kind=kind,
                signature=signature,
                file=file_path,
                line=item.span.start_line + 1 # 1-indexed
            )
        except Exception:  # noqa: BLE001 - one bad symbol must not cost the file
            logger.warning(
                "symbol construction failed for %s in %s (kind=%s, line=%s); skipping THIS "
                "symbol only - the rest of the file is still indexed",
                item.name, file_path, kind, item.span.start_line + 1, exc_info=True,
            )
            skipped_unnameable.append((file_path, item.span.start_line + 1, kind))
            continue

        symbols[f"{item.name}:{kind}"] = s

    if skipped_unnameable:
        # Loud enough to be greppable and countable, per-file, with a number. The old code
        # logged one line per DISCARDED FILE, which read like noise; this reports what was
        # actually lost against what was kept.
        logger.warning(
            "%s: skipped %d un-nameable symbol(s), kept %d - lines %s",
            file_path, len(skipped_unnameable), len(symbols),
            ",".join(str(l) for _, l, _ in skipped_unnameable[:10]),
        )

    return symbols

def _k8s_symbol_delta(file_path: str, old_content: str, new_content: str) -> SymbolDelta:
    """Diff two revisions of a k8s YAML file into a SymbolDelta (board #788).

    Same add/modify/remove shape as the code path, keyed on the resource
    identity, so infra resources flow through the identical writer. Reference
    edges are NOT carried on SymbolDelta — the caller that wants them calls
    ``memex.extractor.k8s.extract_k8s_symbols`` directly and feeds
    ``write_resource_ref_edges``.
    """
    from memex.extractor.k8s import extract_k8s_symbols

    old_syms = extract_k8s_symbols(file_path, old_content).symbols
    new_syms = extract_k8s_symbols(file_path, new_content).symbols

    delta = SymbolDelta()
    for key, new_sym in new_syms.items():
        if key not in old_syms:
            delta.added.append(new_sym)
        elif old_syms[key].signature != new_sym.signature:
            delta.modified.append(new_sym)
    for key, old_sym in old_syms.items():
        if key not in new_syms:
            delta.removed.append(old_sym)
    return delta


async def extract_symbol_delta(
    file_path: str,
    old_content: str,
    new_content: str,
    language: Optional[str] = None,
) -> SymbolDelta:
    if language is None:
        ext = file_path.split(".")[-1]
        # Board #1038: tsx/jsx/mjs/cjs were MISSING, so 320 of smokesignals-web's
        # 775 source files (41%) never reached the extractor at all - they fell
        # through to the "no grammar mapped" branch below and returned an empty
        # delta. Measured coverage before this: 29 of 775 files (3.7%), tsx 0 of 320.
        #
        # .tsx MUST map to the "tsx" grammar, not "typescript": they are separate
        # grammars and parsing JSX with the plain TS grammar silently degrades
        # (it parses, it just does not see the components).
        # .jsx maps to javascript - the JS grammar handles JSX.
        # .mjs/.cjs are plain JavaScript modules.
        lang_map = {
            "py": "python",
            "js": "javascript",
            "jsx": "javascript",
            "mjs": "javascript",
            "cjs": "javascript",
            "ts": "typescript",
            "tsx": "tsx",
            "rs": "rust",
            "go": "go"
        }
        # Board #788 GAP 1: k8s/infra YAML has no functions or classes — it has
        # resources with apiVersion/kind/namespace/name identity. Route it to
        # the dedicated k8s extractor (its own shape, deterministic, no LLM)
        # instead of the tree-sitter code path. Non-k8s YAML yields nothing there
        # (honest), same as an unmapped extension here.
        if ext in ("yaml", "yml"):
            return _k8s_symbol_delta(file_path, old_content, new_content)

        language = lang_map.get(ext)
        if language is None:
            # Council fix 5b: this used to default every unmapped extension
            # (.tf, .hcl, ...) to "python", so e.g. an infrastructure repo's
            # HCL got parsed as Python, produced zero real symbols (a parse
            # mismatch, not an honestly-empty file), and handle_file_change
            # logged it as an ordinary successful index pass. Skip loudly
            # instead of silently guessing wrong. (.yaml/.yml are handled just
            # above; HCL/Terraform remains a separate, later extractor.)
            logger.warning(
                "extract_symbol_delta: no tree-sitter grammar mapped for "
                "extension '.%s' (file=%s) — skipping symbol extraction "
                "for this file instead of silently parsing it as Python",
                ext, file_path,
            )
            return SymbolDelta()

    old_symbols = get_symbols_from_content(old_content, file_path, language)
    new_symbols = get_symbols_from_content(new_content, file_path, language)

    delta = SymbolDelta()

    # Find added and modified
    for key, new_sym in new_symbols.items():
        if key not in old_symbols:
            delta.added.append(new_sym)
        else:
            old_sym = old_symbols[key]
            if old_sym.signature != new_sym.signature:
                delta.modified.append(new_sym)

    # Find removed
    for key, old_sym in old_symbols.items():
        if key not in new_symbols:
            delta.removed.append(old_sym)

    return delta
