"""Board #1038: one un-nameable symbol must NOT destroy a whole file's symbols.

THE BUG THIS PINS: memex/extractor/treesitter.py built Symbol(name=item.name) with
item.name = None for anonymous JS/TS constructs. Symbol.name is a required str, so pydantic
raised. The surrounding try/except wrapped only tslp.process(), NOT the Symbol construction,
so the exception escaped get_symbols_from_content() and extract_symbol_delta() and was caught
in ingest_history.py, which logged a WARNING and DISCARDED THE ENTIRE FILE'S SYMBOLS.

LIVE IMPACT: smokesignals-web ingested 4,773 files and wrote 142 symbols with 0 call edges,
across 734 discarded files, and the job still exited "Complete".

These tests are written to FAIL against the pre-fix code (observed RED before the fix, per the
council's rule that a gate never seen red is not a gate).
"""
import pytest

from memex.extractor.treesitter import get_symbols_from_content

# One named function that MUST survive, sitting beside constructs that yield name=None.
JS_MIXED = """
function keepMe(a) { return a; }
const anonArrow = () => { return 1; };
export default function () { return 2; }
class KeepMeToo { method() {} }
"""


def test_named_symbols_survive_an_unnameable_sibling():
    """The whole point: a nameless construct costs at most ITSELF, never the file."""
    syms = get_symbols_from_content(JS_MIXED, "src/mixed.js", "javascript")
    names = {s.name for s in syms.values()}
    assert "keepMe" in names, (
        "a named function was destroyed by an unnameable sibling in the same file - "
        "this is the #1038 whole-file-loss bug"
    )
    assert "KeepMeToo" in names, "the class was destroyed by the same failure"


def test_extraction_does_not_raise_on_unnameable_constructs():
    """Pre-fix this raises pydantic ValidationError instead of returning."""
    try:
        get_symbols_from_content(JS_MIXED, "src/mixed.js", "javascript")
    except Exception as exc:  # noqa: BLE001 - we are asserting it does NOT happen
        pytest.fail(
            "get_symbols_from_content raised instead of skipping the unnameable symbol: "
            "%s: %s" % (type(exc).__name__, exc)
        )


def test_a_file_of_only_unnameable_constructs_returns_empty_not_raises():
    """Zero symbols is an honest answer. An exception is not - it costs the caller the file."""
    only_anon = "const a = () => 1;\nconst b = () => 2;\n"
    syms = get_symbols_from_content(only_anon, "src/anon.js", "javascript")
    assert isinstance(syms, dict), "must return a mapping even when nothing is nameable"
