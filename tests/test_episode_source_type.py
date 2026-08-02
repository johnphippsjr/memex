"""Board #788 — every add_episode() call must declare source=EpisodeType.text.

graphiti's add_episode defaults to EpisodeType.message, whose extraction prompt
opens with "always extract the speaker (the part before the colon)". Every
episode body this fork writes is descriptive prose about code, and nearly all of
them open with a literal label ("Decision: ...", "Dependency: ...", "Agent
session X started..."), so the default makes the extractor treat that label as a
dialogue participant.

Measured on board #788 over 30 stratified commits, temperature 0, json_object:
source=text won 20 paired commits to 8 with ~1.8x the relationships
(sign test over 28 non-tie pairs, p ~ 0.036).

WHY THIS IS A SOURCE SCAN AND NOT A MOCK: the failure mode is an OMISSION at a
call site. A mock test only covers the call sites someone remembered to write a
mock for, which is exactly the set that would not have been forgotten. This
asserts the PROPERTY over every call site that exists, including ones added
later. Two call sites in tools_write.py were in fact missed on the first pass of
this change and only caught by running this check.
"""

import ast
import pathlib

import pytest

_FORK_ROOT = pathlib.Path(__file__).resolve().parent.parent
_WRITE_SURFACES = [
    "memex/graph/writer.py",
    "memex/graph/cluster_runner.py",
    "memex/mcp_server/tools_write.py",
]


def _add_episode_calls(path: pathlib.Path):
    """Yield (lineno, keywords) for every `*.add_episode(...)` call."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "add_episode":
            yield node.lineno, {kw.arg for kw in node.keywords}


@pytest.mark.parametrize("rel_path", _WRITE_SURFACES)
def test_every_add_episode_declares_source(rel_path):
    """No add_episode call may fall through to the EpisodeType.message default."""
    path = _FORK_ROOT / rel_path
    assert path.exists(), f"write surface moved or renamed: {rel_path}"

    calls = list(_add_episode_calls(path))
    assert calls, f"no add_episode calls found in {rel_path} — did it move?"

    missing = [lineno for lineno, kwargs in calls if "source" not in kwargs]
    assert not missing, (
        f"{rel_path}: add_episode at line(s) {missing} does not pass source=. "
        "It will silently default to EpisodeType.message and be extracted with "
        "the chat-transcript prompt. See board #788."
    )


def test_the_scan_can_actually_fail():
    """Negative control: the checker must FLAG a call that omits source=.

    Without this, a bug in _add_episode_calls (wrong attribute name, wrong AST
    node type) would make every assertion above pass vacuously — a green test
    that checks nothing, which is the failure mode this repo has been bitten by
    before.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        bad = pathlib.Path(d) / "bad.py"
        bad.write_text(
            "async def f(client):\n"
            "    await client.add_episode(name='x', episode_body='y')\n",
            encoding="utf-8",
        )
        calls = list(_add_episode_calls(bad))
        assert len(calls) == 1, "the scanner failed to see an add_episode call at all"
        assert "source" not in calls[0][1], "the scanner failed to notice source= was absent"
