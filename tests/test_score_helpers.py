from bench.score import normalise_text, char_error_rate, text_match, count_match


def test_normalise_maps_ocr_confusables():
    assert normalise_text("0PEN 24 H0URS") == normalise_text("OPEN 24 HOURS")
    assert normalise_text("  Hello,  world!! ") == "HELLO WORLD"


def test_cer():
    assert char_error_rate("OPEN 24 HOURS", "OPEN 24 HOURS") == 0.0
    assert 0 < char_error_rate("OPEN 24 HOUR", "OPEN 24 HOURS") < 0.2
    assert char_error_rate("", "ABC") == 1.0


def test_text_match_substring_and_multiline():
    m, cer = text_match(["CAFE", "OPEN 24 HOURS", "est 1999"], "OPEN 24 HOURS")
    assert m == 1 and cer == 0.0
    m, cer = text_match(["OPFN 24 HOURS"], "OPEN 24 HOURS")
    assert m == 0 and 0 < cer < 0.2
    m, cer = text_match([], "HELLO")
    assert m == 0 and cer == 1.0


def test_count_match():
    assert count_match(3, 3) == 1
    assert count_match(2, 3) == 0


class _Model:
    def __init__(self): self.moves = []
    def to(self, dev): self.moves.append(dev); return self


def test_park_moves_every_loaded_scorer_off_the_gpu():
    from bench.score import Scorers
    sc = Scorers({"scoring": {}})
    sc._pick, sc._clip, sc._det = (_Model(), None, None), (_Model(), None), (_Model(), None)
    sc.park()
    assert [m[0].moves for m in (sc._pick, sc._clip, sc._det)] == [["cpu"]] * 3
    Scorers({"scoring": {}}).park()          # nothing loaded: no-op, no error
