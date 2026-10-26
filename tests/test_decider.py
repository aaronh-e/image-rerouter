import json

from rerouter.decider import QUESTIONS, heuristic_answers, profile_from_answers, route, DEFAULT_POLICY, _flatten


def test_questions_follow_jev_schema():
    for qid, q in QUESTIONS.items():
        assert q["type"] in ("noul", "choice", "score")
        assert q["instructions"]
        if q["type"] == "choice":
            assert isinstance(q["criteria"], dict) and 1 <= len(q["criteria"]) <= 255
        if q["type"] == "score":
            assert isinstance(q["criteria"], list) and 2 <= len(q["criteria"]) <= 10
    json.dumps(QUESTIONS)  # serialisable


def test_heuristic_profiles():
    assert profile_from_answers(heuristic_answers('a storefront sign that says "OPEN 24 HOURS"')) == "text"
    assert profile_from_answers(heuristic_answers("three red cups to the left of a blue plate")) == "layout"
    assert profile_from_answers(heuristic_answers("close-up photo of a hand holding an apple")) == "hands_faces"
    assert profile_from_answers(heuristic_answers("a fox in a snowy forest, watercolor painting")) == "stylized"
    assert profile_from_answers(heuristic_answers("a red ceramic coffee mug with steam rising, plain white background")) == "generic"


def test_route_uses_policy_and_falls_back():
    ans = heuristic_answers('a neon sign reading "HELLO"')
    r = route(ans)
    assert r["direction"] == DEFAULT_POLICY["text"]["direction"] and r["profile"] == "text"
    r = route(ans, {"generic": {"direction": "l2s", "switch_fraction": 0.6}})
    assert r == {"direction": "l2s", "switch_fraction": 0.6, "profile": "text"}


def test_flatten_jev_response_shape():
    raw = {"has_readable_text": {"type": "noul", "noul": 0.93},
           "style": {"type": "choice", "choice": "photoreal", "probabilities": {"photoreal": 0.8, "stylized": 0.1, "unspecified": 0.1}, "confidence": 0.7},
           "detail_density": {"type": "score", "score": 1.4, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}}}
    f = _flatten(raw)
    assert f["has_readable_text"] == 0.93 and f["style"] == "photoreal" and f["detail_density"] == 1.4


def test_route_propagates_steps_from_learned_policy():
    ans = heuristic_answers("a red mug on a table")
    r = route(ans, {"generic": {"direction": "large", "switch_fraction": 0.0, "steps": 18}})
    assert r["direction"] == "large" and r["steps"] == 18
