"""Board #1038: the re-ingest safety preconditions the council made BLOCKING.

Imports memex.graph.writer, which pulls graphiti_core, so this runs in the memex
IMAGE (kubectl exec ... pytest), not on a bare workstation. The extractor/ingest
suites cover what runs without graphiti_core; this covers the graph-write half.

The single most important test here is test_graphiti_accepts_uuid_kwarg: the whole
replayable-episode precondition is a no-op if the installed graphiti-core does not
accept a `uuid` on add_episode. It must PASS in the image before any decisions
re-ingest is allowed to reset a checkpoint.
"""
from memex.graph.writer import (
    deterministic_episode_uuid,
    EXTRACTOR_VERSION,
    ADD_EPISODE_ACCEPTS_UUID,
    reconcile_superseded_symbols,
)


def test_episode_uuid_is_deterministic():
    """Same (repo, commit, episode_name) -> same uuid, so a re-run of a commit
    MERGEs onto the SAME :Episodic node instead of forking a duplicate. This is the
    property the chair called 'safe salvage vs corrupting the crown jewels'."""
    a = deterministic_episode_uuid("repoX", "abc123", "decision_abc123")
    b = deterministic_episode_uuid("repoX", "abc123", "decision_abc123")
    assert a == b


def test_episode_uuid_varies_by_repo_and_commit():
    base = deterministic_episode_uuid("repoX", "abc123", "decision_abc123")
    assert deterministic_episode_uuid("repoY", "abc123", "decision_abc123") != base
    assert deterministic_episode_uuid("repoX", "def456", "decision_def456") != base


def test_deterministic_uuid_is_not_passed_to_add_episode():
    """BOARD #1046: the #1038 replayability idea was WITHDRAWN - graphiti's add_episode(uuid=X)
    is an UPDATE path (get_by_uuid, raises NodeNotFoundError for a new X), so passing a fresh
    deterministic uuid broke EVERY decision write. write_decision must NOT pass `uuid` to
    add_episode. Guard against a well-meaning re-wire by asserting the source no longer does."""
    import inspect
    from memex.graph import writer
    src = inspect.getsource(writer.write_decision)
    assert "uuid" not in src.split("add_episode")[1].split(")")[0], (
        "write_decision passes uuid= to add_episode again - that raises NodeNotFoundError "
        "for every new episode (board #1046). Idempotency comes from the checkpoint, not a uuid."
    )


def test_extractor_version_is_a_nonempty_stamp():
    assert EXTRACTOR_VERSION and isinstance(EXTRACTOR_VERSION, str)


def test_reconcile_defaults_to_dry_run(monkeypatch):
    """reconcile_superseded_symbols must default to dry_run=True (count only): the
    mutating form is destructive-adjacent and only safe after a COMPLETE re-ingest
    from a reset checkpoint. A caller that forgets the flag must NOT mutate."""
    import inspect
    sig = inspect.signature(reconcile_superseded_symbols)
    assert sig.parameters["dry_run"].default is True
