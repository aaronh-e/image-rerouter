import collections
from pathlib import Path

from rerouter.utils import read_prompts, parse_expected

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_has_10_per_group():
    rows = read_prompts(ROOT / "prompts/pilot.csv")
    assert len(rows) == 60
    assert collections.Counter(r["group"] for r in rows) == {g: 10 for g in
        ("composition", "text", "hands_faces", "product", "stylized", "dense")}
    assert len({r["id"] for r in rows}) == 60


def test_expected_is_parseable_where_required():
    for f in ("prompts/pilot.csv", "prompts/full.csv"):
        for r in read_prompts(ROOT / f):
            e = parse_expected(r["expected"])
            if r["group"] == "composition":
                assert isinstance(e.get("count"), int) and e.get("object"), r
            elif r["group"] == "text":
                assert e.get("text") and e["text"] in r["prompt"], r
            else:
                assert e == {}, r


def test_full_is_superset_of_pilot_and_unique():
    pilot = {r["id"]: r["prompt"] for r in read_prompts(ROOT / "prompts/pilot.csv")}
    full = read_prompts(ROOT / "prompts/full.csv")
    ids = [r["id"] for r in full]
    assert len(ids) == len(set(ids))
    for r in full:
        if r["id"] in pilot:
            assert pilot[r["id"]] == r["prompt"]
    assert 190 <= len(full) <= 220


def test_parse_expected_edge_cases():
    assert parse_expected("") == {}
    assert parse_expected(None) == {}
    assert parse_expected("count=3;object=red cup;relation=left of plate") == {"count": 3, "object": "red cup", "relation": "left of plate"}
    assert parse_expected("text=OPEN 24 HOURS") == {"text": "OPEN 24 HOURS"}
