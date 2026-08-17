"""Board #1038: the tags-query extractor must NAME anonymous JS/TS constructs,
must reach .tsx/.jsx/.mjs/.cjs at all, and must NOT turn call sites into definitions.

Every count here was MEASURED against the real queries, not assumed. If a vendored
query file is bumped, these numbers must be re-measured rather than adjusted to pass.
"""
import asyncio

import pytest

from memex.extractor.treesitter import (
    get_symbols_from_content,
    extract_symbol_delta,
    extract_calls,
    _TAG_QUERIES,
)

TSX = """
import React from "react";
const App = () => <div>hi</div>;
export default function Page() { return <App />; }
export const Btn: React.FC = () => <button/>;
const handler = async (e) => { await submit(e); };
const Fancy = forwardRef((props, ref) => <input ref={ref} {...props} />);
const Memoed = React.memo(() => <span/>);
type Props = { a: string };
enum Color { Red, Blue }
class Widget { render() {} }
items.map(x => x + 1);
useEffect(() => { doThing(); }, []);
"""

JS_ANON = """
const arrowConst = () => { return 1; };
let arrowLet = (x) => x * 2;
export const exportedArrow = async (a, b) => a + b;
function namedFn() {}
"""


def _names(src, path, lang):
    return {s.name for s in get_symbols_from_content(src, path, lang).values()}


def test_all_queries_compile_at_import():
    """tree_sitter 0.26.0 poisons a query text for the process lifetime if a compile
    fails, with a misleading error, across languages. So compilation must happen once
    at import - if this module imported, they compiled."""
    assert set(_TAG_QUERIES) == {"javascript", "typescript", "tsx"}


def test_anonymous_js_functions_are_named_by_their_binding():
    """The core #1038 fix: `const foo = () => {}` must be named "foo", not dropped."""
    names = _names(JS_ANON, "src/a.js", "javascript")
    for want in ("arrowConst", "arrowLet", "exportedArrow", "namedFn"):
        assert want in names, "%s was not named - anonymous binding resolution failed" % want


def test_tsx_react_components_are_found():
    """Measured: this returned ZERO with the upstream TypeScript query alone, because
    that query is a SUPPLEMENT to the JavaScript one. JS+TS+supplement returns these."""
    names = _names(TSX, "src/App.tsx", "tsx")
    for want in ("App", "Page", "Btn", "handler", "Widget", "render"):
        assert want in names, "%s missing - the TSX query combination regressed" % want


def test_wrapped_components_are_found():
    """27 forwardRef components in the real repo are invisible without the allowlist."""
    names = _names(TSX, "src/App.tsx", "tsx")
    assert "Fancy" in names, "forwardRef component not named"
    assert "Memoed" in names, "React.memo component not named"


def test_ts_type_level_declarations_are_found():
    names = _names(TSX, "src/App.tsx", "tsx")
    assert "Props" in names, "type alias missing (not in upstream tags.scm)"
    assert "Color" in names, "enum missing (not in upstream tags.scm)"


def test_call_sites_do_NOT_become_definitions():
    """The .captures() vs .matches() bug. On the real repo, .captures() produced
    14,625 reference captures that would have been written as fake definitions."""
    names = _names(TSX, "src/App.tsx", "tsx")
    for bad in ("map", "useEffect", "x", "e", "props", "ref"):
        assert bad not in names, (
            "%r was written as a DEFINITION - call sites/params are leaking in, "
            "which means .captures() semantics crept back" % bad
        )


@pytest.mark.parametrize("path,expect", [
    ("src/App.tsx", "App"),
    ("src/App.jsx", "App"),
    ("src/util.mjs", "App"),
    ("src/util.cjs", "App"),
])
def test_new_extensions_reach_the_extractor(path, expect):
    """41% of the real repo (320 of 775 files) never reached the extractor because
    tsx/jsx/mjs/cjs were absent from lang_map. An unmapped extension returns an
    EMPTY delta, which looks identical to 'this file has no symbols'."""
    src = "const App = () => 1;\n"
    delta = asyncio.run(extract_symbol_delta(path, "", src))
    names = {s.name for s in delta.added}
    assert expect in names, "%s did not reach the extractor (unmapped extension)" % path


# --- board #1038 signature-span fix -----------------------------------------

def test_signature_captures_full_multiline_declaration():
    """The old rule took only the name's FIRST PHYSICAL LINE, so a multi-line
    declaration collapsed to a useless 'function bigFn('. The span rule must carry
    the whole header up to the body."""
    src = "function bigFn(\n  a,\n  b,\n  c,\n) {\n  return a + b + c;\n}\n"
    syms = get_symbols_from_content(src, "src/m.ts", "typescript")
    sig = syms["bigFn:fn"].signature
    assert sig != "function bigFn(", "signature is still just the first physical line"
    for tok in ("bigFn", "a", "b", "c"):
        assert tok in sig, "%r missing from multi-line signature %r" % (tok, sig)


def test_signature_stops_at_body_not_inside_it():
    """A class signature must be the header, not the whole body text."""
    src = "class Widget {\n  render() { return 1; }\n}\n"
    syms = get_symbols_from_content(src, "src/w.tsx", "tsx")
    assert syms["Widget:class"].signature == "class Widget"


# --- board #1038 JS/TS call edges -------------------------------------------

def test_js_ts_call_edges_resolve_arrow_bound_caller():
    """extract_calls was Python-only. It must now resolve calls made INSIDE an
    arrow function bound to a const — the caller tslp.process() could not name."""
    src = ("function helper(x){ return x + 1; }\n"
           "const doWork = (n) => { const r = helper(n); return fmt(r); };\n")
    pairs = {(e.caller, e.callee) for e in extract_calls("src/a.ts", src, language="typescript")}
    assert ("doWork", "helper") in pairs
    assert ("doWork", "fmt") in pairs


def test_module_level_call_fabricates_no_caller():
    """A call at module scope has no enclosing function; we must not invent one."""
    edges = extract_calls("src/b.js", "sideEffect();\n", language="javascript")
    assert edges == []


def test_call_query_language_with_no_query_returns_empty():
    """rust/go extract symbols but have no call query — they must return [] cleanly,
    which is correct (not hollow), never an error."""
    assert extract_calls("src/x.rs", "fn main(){ foo(); }", language="rust") == []
