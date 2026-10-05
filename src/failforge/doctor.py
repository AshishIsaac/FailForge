"""``failforge doctor``: what is installed, what is downloaded, what will run where."""

from __future__ import annotations

import importlib
import json
import platform
import shutil

from failforge import paths


def doctor() -> int:
    ok = True

    def line(name: str, good: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK' if good else '!!'}] {name:<22} {detail}")

    print(f"FailForge doctor - Python {platform.python_version()} on {platform.platform()}")
    print(f"  workspace: {paths.home()}")
    for mod in ("numpy", "cv2", "torch", "torchvision", "ultralytics", "transformers", "diffusers", "sklearn", "umap",
                "fastapi", "matplotlib"):
        try:
            m = importlib.import_module(mod)
            line(mod, True, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001 - report anything
            line(mod, False, f"import failed: {e}")
            ok = False
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            line("GPU", True, f"{p.name}, {p.total_memory / 2**30:.1f} GB, CUDA {torch.version.cuda}")
        else:
            line("GPU", True, "none - CPU mode (training and synthesis will be slow; try the quick profile)")
    except ImportError:
        pass
    m = paths.manifests_dir()
    lock = m / "golden.lock.json"
    if lock.is_file():
        d = json.loads(lock.read_text(encoding="utf-8"))
        line("golden set", True, f"{d['n']} images, frozen {d['created']}, digest {d['digest'][:12]}")
    else:
        line("golden set", False, "not created yet - run `failforge data prepare`")
        ok = False
    n_img = sum(1 for _ in (paths.raw_dir() / "data").glob("*.jpg")) if (paths.raw_dir() / "data").is_dir() else 0
    line("BDD100K images", n_img >= 9000, f"{n_img} on disk")
    hf = paths.models_dir() / "hf" / "hub"
    have = sorted(p.name.replace("models--", "").replace("--", "/") for p in hf.glob("models--*")) if hf.is_dir() else []
    line("model cache", bool(have), ", ".join(have) or "empty - run `failforge models download`")
    free = shutil.disk_usage(paths.home() if paths.home().exists() else paths.REPO).free / 2**30
    line("free disk", free > 5, f"{free:.1f} GB")
    return 0 if ok else 1
