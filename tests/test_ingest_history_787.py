"""Board #787 — DB-free tests for the chronological repo-ingest plumbing.

Everything here runs against a real throwaway git repo built in a tmp dir; no
FalkorDB, no LLM. The end-to-end proof (versioned Symbol nodes, resolved
REFERENCES/CALLS edges, Decision + MOTIVATES, and SymbolEntityType preserving
structural props through a resolve-and-save) was run live against the
memex-fork-test cluster on a throwaway graph — see the board note.
"""

import subprocess
from pathlib import Path

import pytest

from memex.ingest_history import (
    list_commits, changed_files, blob_at, commit_meta, diff_summary, _ext,
    ingest_repo, LIVE_GRAPH_DENYLIST, Checkpoint,
)


def _git(repo, *args):
    subprocess.check_call(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _commit(repo, message):
    # Fixed identity + dates so the tests are deterministic and need no global
    # git config present in the runner.
    env_args = [
        "-c", "user.email=t@example.com", "-c", "user.name=T",
    ]
    subprocess.check_call(
        ["git", "-C", str(repo), *env_args, "commit", "-m", message],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={"GIT_COMMITTER_DATE": "2021-01-01T00:00:00+00:00",
             "GIT_AUTHOR_DATE": "2021-01-01T00:00:00+00:00",
             **_os_environ()},
    )


def _os_environ():
    import os
    return dict(os.environ)


@pytest.fixture()
def repo(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q")
    # commit 1: add a python module with one function
    (r / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    _git(r, "add", "a.py")
    _commit(r, "add foo")
    # commit 2: modify foo's signature (a real Symbol version change)
    (r / "a.py").write_text("def foo(x):\n    return x\n", encoding="utf-8")
    _git(r, "add", "a.py")
    _commit(r, "change foo signature to take x")
    # commit 3: add a k8s manifest, delete nothing
    (r / "d.yaml").write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web\n  namespace: app\n",
        encoding="utf-8",
    )
    _git(r, "add", "d.yaml")
    _commit(r, "add web deployment manifest")
    return r


def test_commits_are_oldest_first(repo):
    shas = list_commits(str(repo))
    assert len(shas) == 3
    # oldest first: commit 1's message is "add foo"
    msg0, _ = commit_meta(str(repo), shas[0])
    assert msg0.strip() == "add foo"
    msg2, _ = commit_meta(str(repo), shas[2])
    assert msg2.strip().startswith("add web deployment")


def test_commit_time_is_the_real_committer_date_not_now(repo):
    shas = list_commits(str(repo))
    _, when = commit_meta(str(repo), shas[0])
    assert when is not None
    assert when.year == 2021  # council fix 4 — not the ingest wall-clock


def test_changed_files_root_commit_is_all_added(repo):
    shas = list_commits(str(repo))
    cfs = changed_files(str(repo), shas[0])
    assert [(c.status, c.path) for c in cfs] == [("A", "a.py")]


def test_changed_files_modify_and_add(repo):
    shas = list_commits(str(repo))
    assert [(c.status, c.path) for c in changed_files(str(repo), shas[1])] == [("M", "a.py")]
    assert [(c.status, c.path) for c in changed_files(str(repo), shas[2])] == [("A", "d.yaml")]


def test_blob_at_reads_content_and_missing_is_empty(repo):
    shas = list_commits(str(repo))
    assert "def foo(x)" in blob_at(str(repo), shas[1], "a.py")
    # d.yaml does not exist at the first commit
    assert blob_at(str(repo), shas[0], "d.yaml") == ""


def test_diff_summary_is_nonempty_and_bounded(repo):
    shas = list_commits(str(repo))
    s = diff_summary(str(repo), shas[0], max_chars=50)
    assert s
    assert len(s) <= 50


def test_ext_helper():
    assert _ext("a/b/c.py") == "py"
    assert _ext("d.YAML") == "yaml"
    assert _ext("Makefile") == ""


def test_live_graph_guard_refuses_without_override(repo):
    """#777 — the ingest must refuse a live graph unless --allow-live. The guard
    is the very first thing ingest_repo does, before any graph client is built,
    so this is DB-free."""
    import asyncio
    live = next(iter(LIVE_GRAPH_DENYLIST))
    with pytest.raises(SystemExit):
        asyncio.run(ingest_repo(repo=str(repo), graph=live, limit=1))


def test_checkpoint_roundtrip(tmp_path):
    p = tmp_path / "cp.txt"
    cp = Checkpoint(p)
    assert not cp.has("abc")
    cp.add("abc")
    cp.add("def")
    # a fresh instance reloads what was flushed
    cp2 = Checkpoint(p)
    assert cp2.has("abc") and cp2.has("def")
    assert not cp2.has("xyz")
