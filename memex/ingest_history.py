"""Board #787 — chronological git-history ingest into a UNIFIED code+memory graph.

Replays a repository's commits oldest-first and, per commit, writes BOTH layers
of the memex graph into the target FalkorDB graph:

  * STRUCTURAL (deterministic, zero-LLM) — for each changed file:
      - a bi-temporal Symbol/resource delta (``write_symbol_delta(versioned=True)``)
        so the graph records what each symbol looked like *at that commit*, not
        just its final shape (board #786);
      - CALLS edges for code files (``write_call_edges``);
      - REFERENCES edges for k8s/infra YAML (``write_resource_ref_edges``, #788).
  * SEMANTIC (LLM) — one commit-granular pass:
      - architectural decisions are synthesised from the commit message + diff
        (``synthesizer.commit.extract_decisions``) and written with their
        MOTIVATES rationale links (``write_decision``, council fix 1 — the
        operator's primary goal), each carrying ``source=EpisodeType.text`` and
        ``entity_types={"Symbol": SymbolEntityType}`` so a resolve-and-save onto
        an existing Symbol node OVERLAYS instead of wiping its structural props.

Everything routes through ``get_graph_client()``, so the LLM path automatically
uses ``SalvagingLocalClient`` (enable_thinking:false + #790 truncation salvage).
Temperature is pinned to 0 via config (#788). The commit's real committer date
anchors every ``valid_from``/``reference_time`` (council fix 4), so replaying
years of history does NOT stamp everything with the ingest wall-clock.

TARGET SAFETY (board #777): this writes into a **test copy**, never the live
graph. ``--graph`` is REQUIRED and a denylist refuses the known live graphs
unless ``--allow-live`` is passed explicitly. Nothing here should ever touch the
live database without a deliberate override.

RESUMABILITY: the versioned Symbol MERGE and the uuid5 Decision identity are
idempotent, but ``add_episode`` is not — so a checkpoint file records processed
commit SHAs and a resumed run skips them, preventing duplicate episodes.

Usage:
    python -m memex.ingest_history --repo /path/to/repo --graph test-787-copy
    python -m memex.ingest_history --repo /path/to/repo --graph test-787-copy \
        --limit 50 --no-decisions          # structural-only smoke test
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Graphs that are NOT test copies. The ingest refuses to write into any of these
# unless --allow-live is passed. mem0-seed-local-verify is the live unified seed
# graph; graphiti-mcp-live is the running MCP's graph. (#777 / operator: do NOT
# ingest into the live database.)
LIVE_GRAPH_DENYLIST = {"mem0-seed-local-verify", "graphiti-mcp-live"}

# Extension -> ("code" | "k8s"). Anything unmapped is still counted but produces
# no structural nodes (mirrors extract_symbol_delta's own honesty contract:
# an unmapped file yields nothing rather than a wrong guess).
# BOARD #1038 - THE SECOND EXTENSION GATE, and the one that actually mattered.
#
# This set is checked at the top of the per-file loop below; anything not in it is
# `continue`d BEFORE extract_symbol_delta is ever called. So fixing the extractor's
# own lang_map (memex/extractor/treesitter.py) was NOT enough on its own - a .tsx
# file never reached the extractor to begin with.
#
# MEASURED on smokesignals-web: 320 of 775 source files are .tsx, and coverage was
# 29/775 = 3.7% with tsx at 0/320. Both this set and the extractor's map had to gain
# the same extensions; fixing one and not the other looks like a working fix and
# silently changes nothing.
#
# NOTE for whoever audits this next: the round-2 council said the second map to fix
# was in memex/watcher/handlers.py. It is NOT - that one feeds extract_calls (call
# edges), which is Python-only by design. THIS is the second gate.
_CODE_EXTS = {"py", "js", "jsx", "mjs", "cjs", "ts", "tsx", "rs", "go"}
_K8S_EXTS = {"yaml", "yml"}

# BOARD #1038 - per-language CAPABILITY table. What each extension is EXPECTED to
# contribute, so a run can ASSERT it got what it should have rather than discover
# a whole language silently produced nothing 13 hours later.
#   "symbols"   -> tree-sitter code symbols (fn/class). A language with files but
#                  zero symbols is HOLLOW (the #1038 defect).
#   "resources" -> k8s resource nodes + REFERENCES edges, NOT code symbols. Zero
#                  code symbols here is CORRECT, so it is deliberately excluded
#                  from the hollow check (alarming on it is how an alert gets muted).
# Keyed identically to _CODE_EXTS / _K8S_EXTS on purpose; the two must not drift.
LANGUAGE_CAPABILITIES = {
    **{ext: "symbols" for ext in _CODE_EXTS},
    **{ext: "resources" for ext in _K8S_EXTS},
}

# BOARD #1038: extension -> language for CALL-edge extraction. Mirrors the
# extractor's own reach (memex/extractor/treesitter.py::_CALL_QUERIES): Python plus
# JS/TS/TSX. rs/go extract SYMBOLS but have no call query yet, so they are absent
# here and correctly produce no call edges - which is NOT a hollow condition, since
# the hollow gate checks symbols, never call edges (the council's point: 0 call
# edges on a language without a call query is correct, not a failure).
_CALL_LANG_BY_EXT = {
    "py": "python",
    "js": "javascript", "jsx": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "tsx": "tsx",
}


# ---------------------------------------------------------------------------
# git plumbing — all read-only, all via subprocess so there is no libgit2 dep.
# Each helper returns "" / [] on any git failure rather than raising, matching
# watcher/git_hook.py's convention (a single unreadable blob must not abort the
# whole multi-thousand-commit replay).
# ---------------------------------------------------------------------------


def _git(repo: str, *args: str, binary: bool = False):
    out = subprocess.check_output(["git", "-C", repo, *args], stderr=subprocess.DEVNULL)
    return out if binary else out.decode("utf-8", errors="ignore")


def list_commits(repo: str, first_parent: bool = True, since: Optional[str] = None) -> List[str]:
    """Return commit SHAs oldest-first (chronological replay order).

    ``first_parent`` walks the mainline only, so a merge's side-branch commits
    are not replayed a second time via the merge; the merge commit's own diff
    (vs its first parent) still carries the resolution. ``since`` (a SHA/ref)
    limits to commits AFTER it — cheap way to ingest a tail for testing.
    """
    rev_args = ["rev-list", "--reverse"]
    if first_parent:
        rev_args.append("--first-parent")
    rev_args.append(f"{since}..HEAD" if since else "HEAD")
    text = _git(repo, *rev_args).strip()
    return [line for line in text.splitlines() if line]


def commit_meta(repo: str, sha: str) -> Tuple[str, Optional[datetime]]:
    """(full message, committer datetime). The datetime is timezone-aware from
    ``%cI`` (strict ISO-8601), parsed with ``fromisoformat``; None if unparseable."""
    message = _git(repo, "log", "-1", "--pretty=%B", sha).strip()
    when: Optional[datetime] = None
    iso = _git(repo, "log", "-1", "--pretty=%cI", sha).strip()
    if iso:
        try:
            when = datetime.fromisoformat(iso)
        except ValueError:
            when = None
    return message, when


def _parent(repo: str, sha: str) -> Optional[str]:
    try:
        return _git(repo, "rev-parse", "--verify", f"{sha}^").strip() or None
    except subprocess.CalledProcessError:
        return None  # root commit


@dataclass
class ChangedFile:
    path: str
    status: str          # "A" | "M" | "D" | "R"
    old_path: Optional[str] = None   # set for renames


def changed_files(repo: str, sha: str) -> List[ChangedFile]:
    """Name-status of a commit vs its first parent (or the empty tree for the
    root commit). Renames (``R``) carry both endpoints so the delta can diff the
    old blob at ``old_path`` against the new blob at ``path``."""
    parent = _parent(repo, sha)
    if parent:
        raw = _git(repo, "diff", "--name-status", "-M", "--first-parent", parent, sha)
    else:
        raw = _git(repo, "show", "--pretty=", "--name-status", "-M", sha)

    files: List[ChangedFile] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        code = parts[0]
        if code.startswith("R") and len(parts) >= 3:
            files.append(ChangedFile(path=parts[2], status="R", old_path=parts[1]))
        elif code.startswith("A") and len(parts) >= 2:
            files.append(ChangedFile(path=parts[1], status="A"))
        elif code.startswith("D") and len(parts) >= 2:
            files.append(ChangedFile(path=parts[1], status="D"))
        elif code.startswith("M") and len(parts) >= 2:
            files.append(ChangedFile(path=parts[1], status="M"))
        # C (copy), T (type change) etc. fall through — rare, and the code/k8s
        # extractors are keyed on the new blob anyway; a missed copy just means
        # one fewer structural delta, never a wrong one.
    return files


def blob_at(repo: str, ref: str, path: str) -> str:
    """Text of ``path`` at ``ref``; "" if it does not exist there or is binary."""
    try:
        data = _git(repo, "show", f"{ref}:{path}", binary=True)
    except subprocess.CalledProcessError:
        return ""
    if b"\x00" in data[:8192]:
        return ""  # binary — the extractors want text
    return data.decode("utf-8", errors="ignore")


def diff_summary(repo: str, sha: str, max_chars: int = 4000) -> str:
    """A compact ``--stat`` summary for the decision synthesiser's prompt.
    Truncated so a huge commit cannot blow the extraction context."""
    try:
        text = _git(repo, "show", "--stat", "--pretty=", sha).strip()
    except subprocess.CalledProcessError:
        return ""
    return text[:max_chars]


def _ext(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


# BOARD #1046: build artifacts committed in historical commits (webpack bundles under dist/,
# vendored node_modules, minified files) must NEVER be extracted - they yield thousands of
# garbage minified symbols. Measured on smokesignals-web full history: 86% of extracted symbols
# (1940 of 2254) came from dist/*.bundle.js. HEAD-populate dodged this only because dist/ is
# gitignored at HEAD; a full-history replay sees the old committed bundles. So exclude them by
# path here, before extraction, for BOTH the ingest_v3 fleet path and the symbols-only history
# path (both call ingest_commit).
_BUILD_ARTIFACT_DIRS = {
    "node_modules", "dist", "build", ".next", ".nuxt", "out", "vendor", "coverage",
    "bower_components", "__pycache__", ".venv", "venv", "site-packages", ".cache",
}
_BUILD_ARTIFACT_SUFFIXES = (
    ".min.js", ".min.mjs", ".min.cjs", ".min.css", ".bundle.js", ".bundle.mjs",
    ".chunk.js", "-min.js",
)


def is_build_artifact(path: str) -> bool:
    """True if `path` is a generated/vendored build artifact that should not be symbol-extracted.
    Matches on a DIRECTORY segment being a known build/dep dir, or the filename being a minified/
    bundled file. Exact-segment match (not substring) so `dist-utils/` or `my-vendor.ts` are safe."""
    norm = path.replace("\\", "/")
    segs = norm.split("/")
    if any(seg in _BUILD_ARTIFACT_DIRS for seg in segs[:-1]):
        return True
    fname = segs[-1].lower()
    return any(fname.endswith(suf) for suf in _BUILD_ARTIFACT_SUFFIXES)


def _detect_intrafile_renames(delta) -> dict:
    """A function renamed IN PLACE (foo->bar, same file, same commit) shows up as
    a removed `foo` + an added `bar` — git tracks file renames, not symbol ones.
    Match them by signature similarity so the stable sid carries (#801).

    Greedy 1:1 above a CONSERVATIVE threshold (difflib ratio >= 0.6); matched
    removals are dropped from ``delta.removed`` so a rename is not ALSO recorded
    as a delete. Returns ``{(new_name,new_file): (old_name,old_file,ratio)}``.
    Heavily-rewritten renames below the threshold are left as delete+add (safe,
    conservative) rather than guessed — the operator was flagged on this default.
    """
    import difflib

    renames: dict = {}
    if not delta.removed or not delta.added:
        return renames
    remaining = list(delta.removed)
    for added in delta.added:
        best, best_ratio = None, 0.0
        for removed in remaining:
            if removed.file != added.file:
                continue
            ratio = difflib.SequenceMatcher(
                None, removed.signature or removed.name, added.signature or added.name
            ).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, removed
        if best is not None and best_ratio >= 0.6:
            renames[(added.name, added.file)] = (best.name, best.file, round(best_ratio, 3))
            remaining.remove(best)
    matched = {(o[0], o[1]) for o in renames.values()}
    delta.removed = [r for r in delta.removed if (r.name, r.file) not in matched]
    return renames


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------


class Checkpoint:
    """A newline-delimited set of processed commit SHAs, flushed after each
    commit so a killed run resumes without re-adding episodes."""

    def __init__(self, path: Path):
        self.path = path
        self.done = set()
        if path.exists():
            self.done = {
                ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
            }

    def has(self, sha: str) -> bool:
        return sha in self.done

    def add(self, sha: str) -> None:
        self.done.add(sha)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(sha + "\n")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@dataclass
class IngestStats:
    commits: int = 0
    skipped: int = 0
    symbols_written: int = 0
    call_edges: int = 0
    ref_edges: int = 0
    decisions: int = 0
    files_seen: int = 0
    errors: int = 0
    # BOARD #1038: per-extension coverage. The whole failure was invisible because
    # the only numbers on offer were TOTALS - 4,773 files_seen and 142
    # symbols_written looks bad only if you already suspect something. Broken down
    # per extension it is unmissable: ts had thousands of files and wrote nothing.
    files_by_ext: dict = field(default_factory=dict)
    symbols_by_ext: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def hollow_extensions(self, min_files: int = 5) -> List[str]:
        """Extensions that had real work to do and produced NOTHING.

        This is the check that would have caught #1038 on the first progress line
        instead of 13 hours later. Deliberately NOT "symbols == 0 is a failure":
          * K8S/YAML is excluded - it flows through the resource extractor, and
            zero code symbols there is CORRECT.
          * min_files guards the honest small case: one .js file with no functions
            in it is not a defect. A LANGUAGE with many files and zero symbols is.
        Returns the offending extensions so the caller can name them, rather than
        a bare bool nobody can act on.
        """
        bad = []
        for ext, n_files in sorted(self.files_by_ext.items()):
            if ext not in _CODE_EXTS:
                continue
            if n_files >= min_files and self.symbols_by_ext.get(ext, 0) == 0:
                bad.append(ext)
        return bad


async def ingest_commit(sha: str, repo: str, repo_id: str, stats: IngestStats,
                        write_decisions: bool = True) -> None:
    """Write one commit's structural + semantic layers. Best-effort per file so a
    single bad blob never aborts the commit."""
    # Imported lazily so --help / the target guard run without a live backend.
    from memex.extractor.treesitter import extract_symbol_delta, extract_calls
    from memex.extractor.k8s import extract_k8s_symbols
    from memex.graph.writer import (
        write_symbol_delta, write_call_edges, write_resource_ref_edges, write_decision,
    )
    from memex.graph.schema import SymbolEntityType

    message, when = commit_meta(repo, sha)
    files = changed_files(repo, sha)
    modules = [f.path for f in files]

    for cf in files:
        stats.files_seen += 1
        ext = _ext(cf.path)
        if ext not in _CODE_EXTS and ext not in _K8S_EXTS:
            continue
        # BOARD #1046: never extract generated/vendored build artifacts (dist/ bundles,
        # node_modules, minified files). Skipped BEFORE the per-ext counters so they do not
        # count toward the hollow check either. Measured: 86% of smokesignals-web full-history
        # symbols were dist/*.bundle.js garbage without this.
        if is_build_artifact(cf.path):
            continue
        # #1038: count per extension so a language that produces nothing is visible
        # in the progress line itself, not only in a post-hoc graph query.
        stats.files_by_ext[ext] = stats.files_by_ext.get(ext, 0) + 1

        old_ref_path = cf.old_path or cf.path
        parent = _parent(repo, sha)
        old_content = "" if cf.status == "A" or not parent else blob_at(repo, parent, old_ref_path)
        new_content = "" if cf.status == "D" else blob_at(repo, sha, cf.path)

        try:
            if cf.status == "R" and cf.old_path:
                # FILE MOVE (git-detected old_path -> path): treat every symbol in
                # the new file as CARRIED from old_path, so its stable sid follows
                # the move instead of orphaning (#801). extract_symbol_delta with
                # old="" yields them all as `added`; the rename map points each at
                # its old path (git similarity ~ exact for a pure rename).
                delta = await extract_symbol_delta(cf.path, "", new_content)
                renames = {(s.name, s.file): (s.name, cf.old_path, 1.0) for s in delta.added}
            else:
                delta = await extract_symbol_delta(cf.path, old_content, new_content)
                renames = _detect_intrafile_renames(delta)
            summary = await write_symbol_delta(
                delta, source_commit=sha, repo_root=repo_id,
                commit_time=when, bitemporal=True, renames=renames,
            )
            _n_syms = (summary or {}).get("symbols", 0)
            stats.symbols_written += _n_syms
            stats.symbols_by_ext[ext] = stats.symbols_by_ext.get(ext, 0) + _n_syms
        except Exception:
            stats.errors += 1
            logger.warning("symbol delta failed for %s @ %s", cf.path, sha[:8], exc_info=True)

        # CALLS edges (board #1038): Python + JS/TS/TSX, matching the extractor's
        # own reach (_CALL_LANG_BY_EXT / treesitter._CALL_QUERIES). rs/go extract
        # symbols but have no call query, so they map to None here and produce no
        # edges - correct, not hollow. write_call_edges resolves conservatively
        # (size(cs)=1), so an unresolved callee is dropped, never mis-linked.
        call_lang = _CALL_LANG_BY_EXT.get(ext)
        if new_content and call_lang:
            try:
                stats.call_edges += await write_call_edges(
                    extract_calls(cf.path, new_content, language=call_lang),
                    repo_root=repo_id,
                )
            except Exception:
                stats.errors += 1
                logger.warning("call edges failed for %s @ %s", cf.path, sha[:8], exc_info=True)

        if new_content and ext in _K8S_EXTS:
            try:
                stats.ref_edges += await write_resource_ref_edges(
                    extract_k8s_symbols(cf.path, new_content), repo_root=repo_id,
                )
            except Exception:
                stats.errors += 1
                logger.warning("ref edges failed for %s @ %s", cf.path, sha[:8], exc_info=True)

    if write_decisions:
        try:
            from memex.synthesizer.commit import extract_decisions
            decisions = await extract_decisions(message, diff_summary(repo, sha), sha)
            for d in decisions:
                await write_decision(
                    d, modules=modules, commit_sha=sha,
                    source="watcher", repo_root=repo_id, commit_time=when,
                    entity_types={"Symbol": SymbolEntityType},
                    searchable_rationale=True,  # #803 council default
                )
                stats.decisions += 1
        except Exception:
            stats.errors += 1
            logger.warning("decision synthesis failed for %s", sha[:8], exc_info=True)


async def ingest_repo(repo: str, graph: str, *, repo_id: Optional[str] = None,
                      since: Optional[str] = None, limit: Optional[int] = None,
                      first_parent: bool = True, write_decisions: bool = True,
                      checkpoint_path: Optional[str] = None,
                      allow_live: bool = False) -> IngestStats:
    """Replay ``repo``'s history into FalkorDB graph ``graph``. Returns IngestStats."""
    if graph in LIVE_GRAPH_DENYLIST and not allow_live:
        raise SystemExit(
            f"refusing to ingest into '{graph}' — that is a LIVE graph (#777: "
            f"ingest a TEST COPY). Pass --allow-live only if you truly mean it."
        )

    # Point the config/driver at the target graph BEFORE anything builds the
    # client. falkor_graph AND unified_group_id must be the SAME string or writes
    # fragment across two physical graphs (see config.py). Reset the cached
    # singleton so a prior import cannot pin the old graph.
    os.environ["FALKOR_GRAPH"] = graph
    os.environ["UNIFIED_GROUP_ID"] = graph
    import memex.config as _cfg
    _cfg._config = None

    from memex.config import canonical_repo_path, resolve_project_id
    resolved_id = repo_id or resolve_project_id(repo) or canonical_repo_path(repo)

    commits = list_commits(repo, first_parent=first_parent, since=since)
    if limit is not None:
        commits = commits[:limit]

    cp_path = Path(checkpoint_path) if checkpoint_path else (
        Path(repo) / ".memex" / f"ingest_checkpoint_{graph}.txt"
    )
    checkpoint = Checkpoint(cp_path)

    stats = IngestStats()
    logger.info("ingest: %d commit(s) from %s -> graph '%s' (repo_id=%s)",
                len(commits), repo, graph, resolved_id)

    for i, sha in enumerate(commits, 1):
        if checkpoint.has(sha):
            stats.skipped += 1
            continue
        await ingest_commit(sha, repo, resolved_id, stats, write_decisions=write_decisions)
        checkpoint.add(sha)
        stats.commits += 1
        if i % 25 == 0 or i == len(commits):
            logger.info("ingest: %d/%d commits (%s)", i, len(commits), json.dumps(stats.as_dict()))

    return stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Board #787 chronological repo ingest into a unified graph.")
    p.add_argument("--repo", required=True, help="path to the git repository to ingest")
    p.add_argument("--graph", required=True, help="target FalkorDB graph (a TEST COPY, #777)")
    p.add_argument("--repo-id", default=None, help="override repo_path identity (default: project_id or canonical path)")
    p.add_argument("--since", default=None, help="only commits AFTER this SHA/ref (tail ingest)")
    p.add_argument("--limit", type=int, default=None, help="cap number of commits (testing)")
    p.add_argument("--all-parents", action="store_true", help="replay every commit, not just --first-parent mainline")
    p.add_argument("--no-decisions", action="store_true", help="structural only — skip LLM decision synthesis")
    p.add_argument("--checkpoint", default=None, help="checkpoint file path (default: <repo>/.memex/ingest_checkpoint_<graph>.txt)")
    p.add_argument("--allow-live", action="store_true", help="permit writing into a denylisted LIVE graph (do not use)")
    return p


def capability_report(stats: IngestStats) -> dict:
    """BOARD #1038 - a per-language coverage line for the run summary / metrics.

    For every extension that had files, report how many symbols it produced and
    whether that is acceptable for its declared capability. This is what turns
    the failure from invisible-in-totals into unmissable-per-language, and it is
    the shape a monitor/metric should push.
    """
    report = {}
    for ext, n_files in sorted(stats.files_by_ext.items()):
        cap = LANGUAGE_CAPABILITIES.get(ext, "unknown")
        n_syms = stats.symbols_by_ext.get(ext, 0)
        report[ext] = {
            "capability": cap,
            "files": n_files,
            "symbols": n_syms,
            # k8s ("resources") is expected to make 0 CODE symbols; only "symbols"
            # languages are held to the produce-something bar.
            "hollow": cap == "symbols" and n_files >= 5 and n_syms == 0,
        }
    return report


def assert_not_hollow(stats: IngestStats, min_files: int = 5) -> List[str]:
    """BOARD #1038 - FAIL LOUDLY ON A HOLLOW RUN. Shared by BOTH runners.

    THIS is the single change that would have caught the original defect. The
    2026-08-16 smokesignals-web run saw 4,773 files, wrote 142 symbols with 0
    call edges across 734 discarded files, and EXITED 0 - so every watcher,
    dashboard and human read it as success. An ingest that had thousands of files
    of a language and produced no symbols for it has not succeeded; it has failed
    quietly, which is worse than failing.

    Lives here (in the image), NOT in either runner's main(), because the
    PRODUCTION neo4j runner is ingest_v3.py (ConfigMap graph-catchup-neo4j-ingest),
    which never calls this module's main(). If this logic lived only in main() it
    would be dead code on the live path - which is exactly what it was until this
    refactor. Both runners now call this one function.

    Raises SystemExit(3) (not 1, so a hollow run is distinguishable from an
    ordinary crash in job logs) when a "symbols" language had >= min_files files
    and produced zero symbols. Returns the offending extensions on the happy path
    (empty list) so a caller can log/metric them without re-deriving.
    """
    hollow = stats.hollow_extensions(min_files=min_files)
    if hollow:
        detail = ", ".join(
            "%s: %d files -> 0 symbols" % (e, stats.files_by_ext.get(e, 0)) for e in hollow
        )
        logger.error(
            "HOLLOW INGEST - extensions produced no symbols despite having files (%s). "
            "The graph for those languages is empty. This is a FAILURE, not a warning: "
            "see board #1038, where exactly this exited 0 and went unnoticed for a day.",
            detail,
        )
        raise SystemExit(3)
    return hollow


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = asyncio.run(ingest_repo(
        repo=args.repo, graph=args.graph, repo_id=args.repo_id, since=args.since,
        limit=args.limit, first_parent=not args.all_parents,
        write_decisions=not args.no_decisions, checkpoint_path=args.checkpoint,
        allow_live=args.allow_live,
    ))
    print(json.dumps(stats.as_dict(), indent=2))
    logger.info("per-language coverage: %s", json.dumps(capability_report(stats)))
    assert_not_hollow(stats)


if __name__ == "__main__":
    main()
