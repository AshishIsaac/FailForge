"""CLIP image / text embeddings (one model serves clustering, cluster naming, drift and QC)."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np

log = logging.getLogger(__name__)


def _features(out):
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, k, None)
        if v is not None:
            return v
    return out[0]


class ClipEmbedder:
    def __init__(self, name: str = "openai/clip-vit-base-patch32", device: str | None = None) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.dtype = dtype
        self.logit_scale = float(self.model.logit_scale.exp().item())

    def images(self, imgs: Iterable[np.ndarray], batch: int = 64) -> np.ndarray:
        """RGB uint8 arrays -> (N, D) L2-normalized float32."""
        out, buf = [], []

        def flush() -> None:
            if not buf:
                return
            px = self.proc(images=buf, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            with self.torch.inference_mode():
                f = _features(self.model.get_image_features(pixel_values=px)).float()
            out.append(self.torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
            buf.clear()

        for im in imgs:
            buf.append(im)
            if len(buf) >= batch:
                flush()
        flush()
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def texts(self, prompts: list[str]) -> np.ndarray:
        tok = self.proc(text=prompts, return_tensors="pt", padding=True, truncation=True)
        tok = {k: v.to(self.device) for k, v in tok.items()}
        with self.torch.inference_mode():
            f = _features(self.model.get_text_features(**tok)).float()
        return self.torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    @property
    def dim(self) -> int:
        return int(self.model.config.projection_dim)

    def zero_shot(self, img_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        """Softmax over prompts, (N, K)."""
        logits = self.logit_scale * img_emb @ text_emb.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)
