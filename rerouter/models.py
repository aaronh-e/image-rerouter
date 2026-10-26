"""Load the two models ONCE and share everything they can share.

SDXL and SSD-1B are both `StableDiffusionXLPipeline`s with the same latent
format (4 channels, scaling_factor 0.13025) and the same scheduler family. So we
build one pipeline from SDXL's text encoders + the fp16-fix VAE, and keep two
UNets on the GPU. Swapping `pipe.unet` is how a handoff happens (see handoff.py).

Memory budget on a 15 GB T4 at fp16:
    SDXL UNet ~5.1 GB + SSD-1B UNet ~2.7 GB + two text encoders ~1.6 GB
    + VAE 0.16 GB  ->  ~9.6 GB of weights, leaving ~5 GB for activations.
If that is too tight, set compute.attention_slicing: true in config.yaml, or
enable VAE tiling (bank.pipe.vae.enable_tiling()).

torch/diffusers are imported lazily so the rest of the repo imports on a CPU box.
"""
from __future__ import annotations

import gc
import time
from typing import Optional


def pick_device(preferred: str | None = None) -> str:
    """cuda (NVIDIA) -> mps (Apple Silicon) -> cpu. CPU works but is very slow:
    SDXL at 1024x1024 takes many minutes per image; use scripts/level1_local.py
    with a smaller size for a demo."""
    import torch
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, cfg_dtype: str = "float16"):
    import torch
    if device == "cpu":
        return torch.float32          # fp16 matmuls are unsupported/slow on CPU
    return getattr(torch, cfg_dtype)


def apply_memory_options(pipe, compute_cfg: dict) -> None:
    """Attention slicing is a pipeline method, but VAE slicing/tiling live on the VAE
    itself: StableDiffusionXLPipeline has no enable_vae_slicing (only the SD 1.x
    pipeline does), which crashed the first real load on Colab."""
    if compute_cfg.get("attention_slicing"):
        pipe.enable_attention_slicing()
    if compute_cfg.get("vae_slicing", True):
        pipe.vae.enable_slicing()


class ModelBank:
    """Holds one pipeline plus a dict of UNets keyed by tier name ("large", "small", "tiny")."""

    def __init__(self, cfg: dict, device: str = "auto"):
        self.cfg = cfg
        self.device = pick_device(device)
        self.pipe = None            # StableDiffusionXLPipeline (text encoders + VAE + scheduler)
        self.img2img = None         # built on demand for the "twopipe" implementation
        self.unets: dict = {}       # tier name -> UNet2DConditionModel
        self.gpu_name = "cpu"
        self.load_seconds: dict = {}

    # ------------------------------------------------------------ loading --

    def load(self, tiers=("large", "small")) -> "ModelBank":
        import torch
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel

        dtype = pick_dtype(self.device, self.cfg["generation"].get("dtype", "float16"))
        self.dtype = dtype
        m = self.cfg["models"]
        if self.device == "cuda":
            self.gpu_name = torch.cuda.get_device_name(0)
        elif self.device == "mps":
            self.gpu_name = "Apple MPS"
        else:
            self.gpu_name = "cpu"

        t0 = time.time()
        # fp16-fix VAE: the stock SDXL VAE overflows to NaN (black images) in fp16.
        vae = AutoencoderKL.from_pretrained(m["vae"], torch_dtype=dtype)

        # The pipeline "shell": text encoders, tokenizers, scheduler, VAE. Its UNet
        # is whichever tier `text_encoders_from` names (default: large).
        base_id = m[m.get("text_encoders_from", "large")]
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_id, vae=vae, torch_dtype=dtype, variant="fp16", use_safetensors=True)
        # Same scheduler object for every run (fairness rule). Both models ship
        # EulerDiscrete with identical config, so this is a no-op copy that pins it.
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(self.device)
        apply_memory_options(self.pipe, self.cfg["compute"])
        self.unets[m.get("text_encoders_from", "large")] = self.pipe.unet
        self.load_seconds["pipeline"] = time.time() - t0

        for tier in tiers:
            if tier in self.unets:
                continue
            t0 = time.time()
            unet = UNet2DConditionModel.from_pretrained(
                m[tier], subfolder="unet", torch_dtype=dtype, variant="fp16", use_safetensors=True)
            unet.to(self.device).eval()
            self.unets[tier] = unet
            self.load_seconds[tier] = time.time() - t0

        self._original_unet = self.pipe.unet
        self._check_unets_agree()
        return self

    def _check_unets_agree(self) -> None:
        """A UNet swap is only valid if every UNet expects the same inputs. Fail
        loudly here instead of producing garbage silently (the 'subtle' failure
        mode in the research plan)."""
        ref = next(iter(self.unets.values())).config
        for name, u in self.unets.items():
            for key in ("in_channels", "out_channels", "cross_attention_dim", "addition_time_embed_dim",
                        "projection_class_embeddings_input_dim"):
                if u.config.get(key) != ref.get(key):
                    raise RuntimeError(f"UNet '{name}' disagrees on {key}: {u.config.get(key)} vs {ref.get(key)}")

    # ------------------------------------------------------------ helpers --

    def set_unet(self, tier: str) -> None:
        self.pipe.unet = self.unets[tier]

    def restore(self) -> None:
        self.pipe.unet = self._original_unet

    def get_img2img(self):
        """Img2img pipeline sharing all weights with `self.pipe` (used by impl=twopipe)."""
        if self.img2img is None:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            self.img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipe)
            self.img2img.set_progress_bar_config(disable=True)
        return self.img2img

    def param_counts(self) -> dict:
        return {k: sum(p.numel() for p in u.parameters()) / 1e9 for k, u in self.unets.items()}

    def free(self) -> None:
        """Release GPU memory before loading other models (e.g. SD 1.5 in the notebook)."""
        import torch
        self.unets.clear()
        self.pipe = None
        self.img2img = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def free_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
