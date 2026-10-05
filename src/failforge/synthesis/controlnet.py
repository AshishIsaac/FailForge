"""Label-preserving image-to-image synthesis with Stable Diffusion 1.5 + ControlNet (canny + depth).

The trick that makes generative data *labeled* data: we never generate a scene from scratch. We
take a labeled daytime photo and re-render it under the target condition with img2img, while two
ControlNets pin its structure, Canny edges (object outlines) and a monocular depth map (layout).
Objects stay where they were, so the source's boxes are the generated image's boxes. The quality
filters (quality.py) then verify that per box instead of trusting it.

Memory: SD1.5 fp16 + two ControlNets fit a 6 GB laptop GPU with model CPU offload (default).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from failforge.config import SynthesisConfig

log = logging.getLogger(__name__)


def canny_map(rgb: np.ndarray, lo: int = 80, hi: int = 180) -> np.ndarray:
    e = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), lo, hi)
    return np.repeat(e[..., None], 3, axis=2)


class DepthEstimator:
    def __init__(self, name: str, device: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            d = self.model(**inp).predicted_depth[0].float().cpu().numpy()
        d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)  # relative inverse depth, near = bright
        g = (d * 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)


class ControlNetGenerator:
    def __init__(self, cfg: SynthesisConfig, device: str | None = None) -> None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler

        self.cfg = cfg
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        variant = "fp16"  # only the fp16 files are downloaded; on CPU they are upcast to fp32 at load time
        nets = [_load_controlnet(ControlNetModel, n, dtype, variant) for n in cfg.controlnets]
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            cfg.base_model, controlnet=nets if len(nets) > 1 else nets[0], torch_dtype=dtype, variant=variant,
            safety_checker=None, requires_safety_checker=False)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        if self.device == "cuda" and cfg.cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)
        self.pipe.vae.enable_slicing() if hasattr(self.pipe.vae, "enable_slicing") else None
        self.depth = DepthEstimator(cfg.depth_model, self.device) if any("depth" in n for n in cfg.controlnets) else None

    def controls(self, rgb_small: np.ndarray) -> list[np.ndarray]:
        out = []
        for n in self.cfg.controlnets:
            if "canny" in n:
                out.append(canny_map(rgb_small))
            elif "depth" in n:
                out.append(self.depth(rgb_small))
            else:
                raise ValueError(f"no control-map builder for {n}")
        return out

    def generate(self, rgb: np.ndarray, prompt: str, negative: str, seed: int, condition: str = "night"
                 ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Returns (generated RGB at the source resolution, control maps)."""
        from PIL import Image

        from failforge.synthesis.augment import relight

        c = self.cfg
        small = cv2.resize(rgb, (c.width, c.height), interpolation=cv2.INTER_AREA)
        ctrl = self.controls(small)  # structure always comes from the original photo
        init = relight(small, condition, np.random.default_rng(seed)) if c.relit_init else small
        g = self.torch.Generator(device="cpu").manual_seed(seed)
        ctrl_imgs = [Image.fromarray(x) for x in ctrl]
        res = self.pipe(
            prompt=prompt, negative_prompt=negative, image=Image.fromarray(init),
            control_image=ctrl_imgs if len(ctrl_imgs) > 1 else ctrl_imgs[0],
            controlnet_conditioning_scale=c.control_scales if len(c.control_scales) > 1 else c.control_scales[0],
            strength=c.strength, num_inference_steps=c.steps, guidance_scale=c.guidance,
            width=c.width, height=c.height, generator=g)
        out = np.asarray(res.images[0].convert("RGB"))
        out = cv2.resize(out, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        return out, ctrl


def _load_controlnet(cls, name: str, dtype, variant):
    try:
        return cls.from_pretrained(name, torch_dtype=dtype, variant=variant)
    except (OSError, ValueError):  # repo without an fp16 variant
        return cls.from_pretrained(name, torch_dtype=dtype)


def download_models(cfg: SynthesisConfig) -> None:
    """Fetch the diffusion weights into the Hugging Face cache (used by first-run setup)."""
    from huggingface_hub import snapshot_download

    snapshot_download(cfg.base_model, allow_patterns=[
        "model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json", "text_encoder/*.fp16.safetensors",
        "unet/config.json", "unet/*.fp16.safetensors", "vae/config.json", "vae/*.fp16.safetensors",
        "feature_extractor/*"])
    for n in cfg.controlnets:
        snapshot_download(n, allow_patterns=["config.json", "*.fp16.safetensors"])
    snapshot_download(cfg.depth_model)
