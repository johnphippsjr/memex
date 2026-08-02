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
_CODE_EXTS = {"py", "js", "ts", "rs", "go"}
_K8S_EXTS = {"yaml", "yml"}


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

    def as_dict(self) -> dict:
        return self.__dict__.copy()


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
            stats.symbols_written += (summary or {}).get("symbols", 0)
        except Exception:
            stats.errors += 1
            logger.warning("symbol delta failed for %s @ %s", cf.path, sha[:8], exc_info=True)

        # CALLS edges are Python-only in the extractor (_CALL_QUERIES ships
        # python only; other grammars differ per language). Symbols are still
        # extracted for js/ts/rs/go via extract_symbol_delta above — only the
        # call graph is python-scoped here, matching the extractor's own reach.
        if new_content and ext == "py":
            try:
                stats.call_edges += await write_call_edges(
                    extract_calls(cf.path, new_content, language="python"),
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


if __name__ == "__main__":
    main()
