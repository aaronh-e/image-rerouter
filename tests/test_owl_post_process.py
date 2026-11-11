"""The counter must call OWLv2 post-processing through APIs both transformers 4 and 5
have. Seen on Colab with transformers 5: "'Owlv2Processor' object has no attribute
'post_process_object_detection'" on the first image, so count_match stayed empty."""
from unittest.mock import MagicMock

from bench.score import owl_post_process


class FakeOwlv2Processor:
    """Only what transformers 5's Owlv2Processor exposes for this."""
    def __init__(self):
        self.image_processor = MagicMock(spec=["post_process_object_detection"])
        self.image_processor.post_process_object_detection.return_value = [{"boxes": "b", "scores": "s"}]
    # deliberately no post_process_object_detection on the processor itself


def test_post_process_goes_through_the_image_processor():
    proc = FakeOwlv2Processor()
    res = owl_post_process(proc, "outputs", 0.3, "sizes")
    assert res == {"boxes": "b", "scores": "s"}
    proc.image_processor.post_process_object_detection.assert_called_once_with("outputs", threshold=0.3, target_sizes="sizes")
