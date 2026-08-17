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
