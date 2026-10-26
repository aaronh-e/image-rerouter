from rerouter.schedule import plan, plan_from_fractions, steps_for_first, describe


def test_fractions_round_to_whole_steps():
    assert steps_for_first(0.4, 30) == 12
    assert steps_for_first(0.2, 30) == 6
    assert steps_for_first(0.33, 30) == 10


def test_s2l_and_l2s_orders():
    s = plan("s2l", 0.4, 30)
    assert [(x.model, x.start, x.end) for x in s] == [("small", 0, 12), ("large", 12, 30)]
    l = plan("l2s", 0.4, 30)
    assert [(x.model, x.start, x.end) for x in l] == [("large", 0, 12), ("small", 12, 30)]


def test_endpoints_collapse_to_controls():
    assert plan("s2l", 0.0, 30) == plan("large", 0.0, 30)
    assert plan("s2l", 1.0, 30) == plan("small", 0.0, 30)
    assert plan("l2s", 0.0, 30) == plan("small", 0.0, 30)
    assert plan("l2s", 1.0, 30) == plan("large", 0.0, 30)


def test_sandwich():
    s = plan_from_fractions(["large", "small", "large"], [0.4, 0.8], 30)
    assert describe(s) == "large[0:12] -> small[12:24] -> large[24:30]"
    # adjacent same-model segments merge
    s = plan_from_fractions(["large", "large", "small"], [0.2, 0.5], 30)
    assert describe(s) == "large[0:15] -> small[15:30]"


def test_total_steps_preserved():
    for d in ("s2l", "l2s"):
        for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            assert sum(x.n_steps for x in plan(d, f, 30)) == 30
