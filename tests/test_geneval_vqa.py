"""GenEval rules, the paper prompt file, the GenEval tab, and making GPU room for VQAScore."""
import csv
import json
import sys
import types
from pathlib import Path

from bench.score import geneval_judge, nms_keep, relative_position
from rerouter.utils import parse_expected, read_prompts

from test_app_quality import load_app, no_progress

ROOT = Path(__file__).resolve().parents[1]


def no_colors(cls, boxes):
    raise AssertionError("colour asked for a prompt without a colour")


def test_relative_position_matches_geneval():
    a, b = [0, 40, 20, 60], [80, 40, 100, 60]           # a far left of b, same height
    assert relative_position(a, b) == {"left of"}
    assert relative_position(b, a) == {"right of"}
    assert relative_position([40, 0, 60, 20], [40, 80, 60, 100]) == {"above"}
    assert relative_position([0, 0, 50, 50], [5, 5, 55, 55]) == set()   # overlapping: no relation


def test_counting_needs_exactly_n():
    meta = {"tag": "counting", "include": [{"class": "clock", "count": 2}], "exclude": [{"class": "clock", "count": 3}]}
    box = [0, 0, 10, 10]
    assert geneval_judge(meta, {"clock": [box] * 2}, no_colors) == (True, "")
    ok, why = geneval_judge(meta, {"clock": [box]}, no_colors)
    assert not ok and "found 1" in why
    ok, why = geneval_judge(meta, {"clock": [box] * 3}, no_colors)
    assert not ok and "clock<3" in why


def test_color_and_position_rules():
    meta = {"tag": "colors", "include": [{"class": "car", "count": 1, "color": "blue"}]}
    assert geneval_judge(meta, {"car": [[0, 0, 9, 9]]}, lambda c, b: ["blue"])[0]
    ok, why = geneval_judge(meta, {"car": [[0, 0, 9, 9]]}, lambda c, b: ["red"])
    assert not ok and "found red" in why

    meta = {"tag": "position", "include": [{"class": "bear", "count": 1},
                                           {"class": "dog", "count": 1, "position": ["right of", 0]}]}
    left, right = [0, 40, 20, 60], [80, 40, 100, 60]
    assert geneval_judge(meta, {"bear": [left], "dog": [right]}, no_colors)[0]
    ok, why = geneval_judge(meta, {"bear": [right], "dog": [left]}, no_colors)
    assert not ok and "left of" in why
    ok, why = geneval_judge(meta, {"dog": [right]}, no_colors)            # no bear: no target
    assert not ok and "no target" in why


def test_nms_keep_returns_most_confident_first():
    boxes = [[0, 0, 10, 10], [50, 50, 60, 60], [0, 0, 10, 11]]
    assert nms_keep(boxes, [0.5, 0.9, 0.7], iou=0.5) == [1, 2]


def test_paper_prompts_are_ours_plus_geneval():
    rows = read_prompts(ROOT / "prompts/paper.csv")
    full = read_prompts(ROOT / "prompts/full.csv")
    assert rows[:len(full)] == full
    gen = rows[len(full):]
    assert len(gen) == 553 and len({r["id"] for r in rows}) == len(rows)
    for r in gen:
        meta = json.loads(parse_expected(r["expected"])["geneval"])
        assert r["group"] == f"geneval_{meta['tag']}" and meta["prompt"] == r["prompt"]


def test_score_frame_fills_vqa_and_geneval(tmp_path):
    import pandas as pd
    from bench.score import score_frame
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    df = pd.DataFrame([{"prompt_id": "geneval_000", "prompt": "a photo of a bench", "image_path": str(img), "notes": ""}])
    meta = {"tag": "single_object", "include": [{"class": "bench", "count": 1}], "prompt": "a photo of a bench"}
    out = score_frame(df, {"scoring": {}}, ["vqa", "geneval"], fake=True, force=False,
                      expected_lookup={"geneval_000": {"geneval": json.dumps(meta)}})
    assert 0 <= out.at[0, "vqa_score"] <= 1 and out.at[0, "geneval_pass"] in (0, 1)


def test_geneval_tab_reports_cost_and_pass_rates(tmp_path):
    app = load_app(tmp_path)
    methods = [list(app.METHODS)[0], "Large model only"]
    gallery, summary, rows, note = app.run_geneval(42, 30, 1024, 7.5, methods, 0.4, 2, progress=no_progress)
    assert len(rows) == len(gallery) == 6 * 2 * 2            # 6 tasks x 2 prompts x 2 methods
    assert {r[0] for r in rows} == set(app.GENEVAL_TASKS)
    assert all(r[6] in ("pass", "fail") for r in rows)
    assert [s[0] for s in summary] == methods
    header = ["run", "images", "GPU-seconds (mean)", "peak GB", "$/1000 images", "vs large", "GenEval score", *app.GENEVAL_TASKS]
    assert all(len(s) == len(header) for s in summary)
    s2l, large = summary
    assert s2l[1] == 12 and s2l[4] > 0 and s2l[5].startswith("+") and large[5] == "+0%"
    assert 0 <= s2l[6] <= 1
    logged = list(csv.DictReader(open(tmp_path / "geneval.csv")))
    assert len(logged) == 24


def test_geneval_tab_shares_the_gpu_queue(tmp_path):
    app = load_app(tmp_path)
    fns = {f.name: f for f in app.build().fns.values()}
    assert fns["run_geneval"].concurrency_id == fns["run_single"].concurrency_id
    assert fns["run_geneval"].concurrency_limit == 1


def test_image_models_leave_a_small_gpu_while_scoring(tmp_path, monkeypatch):
    app = load_app(tmp_path)
    moves = []
    bank = types.SimpleNamespace(device="cuda", move=moves.append)
    app.STATE.update(demo=False, bank=bank)
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=15e9)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    app.scoring_room()()
    assert moves == ["cpu", "cuda"]
    moves.clear()
    fake_torch.cuda.get_device_properties = lambda i: types.SimpleNamespace(total_memory=80e9)
    app.scoring_room()()
    assert moves == []                                       # big GPU: everything fits, no moving


def test_old_log_is_set_aside_not_mixed(tmp_path):
    app = load_app(tmp_path)
    log = tmp_path / "scores.csv"
    log.write_text("image,prompt,pick_score\nold.png,x,20\n")
    app.append_row(log, ["image", "vqa_score"], ["new.png", 0.9])
    assert log.read_text().splitlines() == ["image,vqa_score", "new.png,0.9"]
    assert len(list(tmp_path.glob("scores.*.old.csv"))) == 1


def test_park_includes_the_vqa_model():
    from bench.score import Scorers

    class M:
        moves = []
        def to(self, dev): self.moves.append(dev); return self

    sc = Scorers({"scoring": {}})
    sc._vqa = (M(), None, 0)
    sc.park()
    assert M.moves == ["cpu"]
