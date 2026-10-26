"""The website's quality checks use the benchmark's scorers (demo mode = fake scores, no GPU)."""
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_app(tmp_path):
    spec = importlib.util.spec_from_file_location("rerouter_app", ROOT / "app" / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.STATE.update(demo=True, bank=None)
    app.OUT = tmp_path
    return app


def no_progress(*a, **k):
    pass


def test_single_scores_every_image_and_logs_them(tmp_path):
    app = load_app(tmp_path)
    gallery, rows, note = app.run_single('a sign that says "OPEN"', 42, 30, 1024, 7.5,
                                         [list(app.METHODS)[0], "Large model only", "Small model only"], 0.4,
                                         True, "OPEN", "", 0, progress=no_progress)
    assert len(rows) == 3
    header = ["run", "plan", "GPU-seconds", "peak GB", "$/1000 images", "file", "vs large", "PickScore", "CLIPScore", "OCR text", "count"]
    assert all(len(r) == len(header) for r in rows)
    for r in rows:
        assert r[4] > 0                                    # cost in dollars
        assert r[7] != "" and r[8] != ""                   # PickScore, CLIPScore
        assert r[9].startswith(("yes", "no"))              # OCR checked because expected text was given
        assert r[10] == ""                                 # no count asked for
    logged = list(csv.DictReader(open(tmp_path / "scores.csv")))
    assert [l["image"] for l in logged] == [r[5] for r in rows]


def test_count_check_and_scoring_off(tmp_path):
    app = load_app(tmp_path)
    _, rows, _ = app.run_single("three red cups", 1, 30, 1024, 7.5, ["Large model only"], 0.0,
                                True, "", "cup", 3, progress=no_progress)
    assert rows[0][9] == "" and rows[0][10].endswith("/3")
    _, rows, _ = app.run_single("three red cups", 1, 30, 1024, 7.5, ["Large model only"], 0.0,
                                False, "", "cup", 3, progress=no_progress)
    assert rows[0][7:] == ["", "", "", ""]


def test_pick_any_mix_of_methods(tmp_path):
    app = load_app(tmp_path)
    _, rows, _ = app.run_single("an apple", 42, 30, 1024, 7.5, list(app.METHODS), 0.4,
                                False, "", "", 0, progress=no_progress)
    # one image per ticked box, always in the same order, each with its own plan
    assert [r[0] for r in rows] == list(app.METHODS)
    assert [r[1] for r in rows] == ["small[0:12] -> large[12:30]", "large[0:12] -> small[12:30]", "large[0:30]", "small[0:30]"]
    assert all(r[6] for r in rows)                         # 'vs large' filled because large-only was ticked
    _, rows, _ = app.run_single("an apple", 42, 30, 1024, 7.5, [list(app.METHODS)[1]], 0.4,
                                False, "", "", 0, progress=no_progress)
    assert len(rows) == 1 and rows[0][6] == ""             # no large-only baseline to compare against


def test_no_method_ticked_is_an_error(tmp_path):
    import gradio as gr
    import pytest
    app = load_app(tmp_path)
    with pytest.raises(gr.Error):
        app.run_single("an apple", 42, 30, 1024, 7.5, [], 0.4, False, "", "", 0, progress=no_progress)


def test_sweep_rows_carry_cost_and_scores(tmp_path):
    app = load_app(tmp_path)
    _, rows, _ = app.run_sweep("an apple", 42, 30, 1024, 7.5, "0.4", True, True, "", "", 0, progress=no_progress)
    header = ["setting", "plan", "GPU-seconds", "$/1000 images", "vs large", "PickScore", "CLIPScore", "OCR text", "count"]
    assert len(rows) == 4 and all(len(r) == len(header) for r in rows)
    assert all(r[5] != "" for r in rows)


def test_failed_scorer_says_unavailable_not_blank(tmp_path):
    app = load_app(tmp_path)

    class Broken:
        disabled = {"pick_score": "out of memory", "ocr": "boom"}
        def pick_score(self, path, prompt): return None
        def clip_score(self, path, prompt): return 28.5
        def ocr(self, path, text): return None, None, None
        def count(self, path, obj, n): return 1, 3
        def park(self): pass

    app.STATE["scorers"] = Broken()
    _, rows, note = app.run_single("three cups", 1, 30, 1024, 7.5, ["Large model only"], 0.0,
                                   True, "OPEN", "cup", 3, progress=no_progress)
    assert rows[0][7:] == ["unavailable", 28.5, "unavailable", "3/3"]
    assert "pick_score" in note and "ocr" in note


def test_generate_and_sweep_share_one_gpu_queue(tmp_path):
    # Both buttons swap UNets on the one shared pipeline, so they must never run at the same time.
    app = load_app(tmp_path)
    fns = {f.name: f for f in app.build().fns.values()}
    a, b = fns["run_single"], fns["run_sweep"]
    assert a.concurrency_id == b.concurrency_id
    assert a.concurrency_limit == b.concurrency_limit == 1


def test_blank_seed_is_a_clear_error(tmp_path):
    import gradio as gr
    import pytest
    app = load_app(tmp_path)
    with pytest.raises(gr.Error):
        app.run_single("an apple", None, 30, 1024, 7.5, ["Large model only"], 0.0, False, "", "", 0, progress=no_progress)
    with pytest.raises(gr.Error):
        app.run_sweep("an apple", None, 30, 1024, 7.5, "0.4", True, False, "", "", 0, progress=no_progress)


def test_file_names_never_collide(tmp_path):
    app = load_app(tmp_path)
    for _ in range(2):   # same settings twice within the same second
        app.run_single("an apple", 42, 30, 1024, 7.5, ["Large model only"], 0.0, False, "", "", 0, progress=no_progress)
    assert len(list(tmp_path.glob("*.png"))) == 2


def test_auth_parsing():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rerouter_app2", ROOT / "app" / "app.py")
    app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
    assert app.parse_auth("") is None and app.parse_auth(None) is None
    assert app.parse_auth("aaron:s3cret:x") == ("aaron", "s3cret:x")
    import pytest
    with pytest.raises(SystemExit):
        app.parse_auth("no-colon")


def test_scoring_runs_after_all_generation_then_parks(tmp_path):
    # Scorers must never sit on the GPU while an image is being generated (T4 OOM).
    app = load_app(tmp_path)
    events = []
    real_gen = app.gen
    def gen(*a, **k):
        events.append("gen"); return real_gen(*a, **k)
    app.gen = gen

    class Rec:
        disabled = {}
        def pick_score(self, p, t): events.append("score"); return 20.0
        def clip_score(self, p, t): return 30.0
        def ocr(self, p, t): return 1, 0.0, t
        def count(self, p, o, n): return 1, n
        def park(self): events.append("park")

    app.STATE["scorers"] = Rec()
    app.run_single("an apple", 1, 30, 1024, 7.5, list(app.METHODS), 0.4, True, "", "", 0, progress=no_progress)
    assert events == ["gen"] * 4 + ["score"] * 4 + ["park"]
    events.clear()
    app.run_sweep("an apple", 1, 30, 1024, 7.5, "0.4", True, True, "", "", 0, progress=no_progress)
    assert events == ["gen"] * 4 + ["score"] * 4 + ["park"]
