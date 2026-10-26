"""The memory options must be applied through APIs that StableDiffusionXLPipeline
actually has. The SDXL pipeline has no enable_vae_slicing (only the SD 1.x pipeline
does); VAE slicing/tiling live on the VAE itself. Seen as
"'StableDiffusionXLPipeline' object has no attribute 'enable_vae_slicing'" on Colab."""
from unittest.mock import MagicMock

from rerouter.models import apply_memory_options


class FakeSDXLPipe:
    """Only what the real pipeline class exposes."""
    def __init__(self):
        self.vae = MagicMock(spec=["enable_slicing", "enable_tiling"])
        self.enable_attention_slicing = MagicMock()
    # deliberately no enable_vae_slicing / enable_vae_tiling


def test_vae_slicing_goes_through_the_vae():
    pipe = FakeSDXLPipe()
    apply_memory_options(pipe, {"vae_slicing": True, "attention_slicing": False})
    pipe.vae.enable_slicing.assert_called_once()
    pipe.enable_attention_slicing.assert_not_called()


def test_attention_slicing_and_off_switches():
    pipe = FakeSDXLPipe()
    apply_memory_options(pipe, {"vae_slicing": False, "attention_slicing": True})
    pipe.enable_attention_slicing.assert_called_once()
    pipe.vae.enable_slicing.assert_not_called()
