"""Board #790 — the truncated-response salvage in the fork's local client."""

from memex.graph.client import _salvage_truncated_json


def test_valid_json_passes_through_unchanged():
    assert _salvage_truncated_json('{"edges": [{"a": 1}]}') == {"edges": [{"a": 1}]}


def test_truncated_mid_number_is_repaired():
    """The counting-loop shape: an object whose array of numbers is cut mid-run.
    The recovered object keeps the complete elements and closes the structures."""
    raw = '{"edges": [{"source": "A", "fact": "x", "episode_indices": [0, 1, 2, 3, 4'
    out = _salvage_truncated_json(raw)
    assert out is not None
    assert out["edges"][0]["source"] == "A"
    assert out["edges"][0]["fact"] == "x"
    # the runaway list keeps the complete integers it had
    assert out["edges"][0]["episode_indices"] == [0, 1, 2, 3, 4]


def test_multiple_complete_edges_before_a_runaway_all_survive():
    raw = ('{"edges": ['
           '{"s": "A", "episode_indices": [0, 1]}, '
           '{"s": "B", "episode_indices": [0, 1]}, '
           '{"s": "C", "episode_indices": [0, 1, 2, 3, 4, 5, 6')
    out = _salvage_truncated_json(raw)
    assert out is not None
    assert [e["s"] for e in out["edges"]] == ["A", "B", "C"]


def test_truncated_inside_a_string_does_not_crash():
    raw = '{"edges": [{"fact": "an unterminated string that got cut'
    out = _salvage_truncated_json(raw)
    # either recovers something valid or returns None — never raises
    assert out is None or isinstance(out, dict)


def test_unrecoverable_returns_none_not_an_exception():
    assert _salvage_truncated_json("not json at all {[") is None
