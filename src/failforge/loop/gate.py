"""Promotion gate: may a candidate model replace the champion?

All checks are on the frozen golden set, against the champion, using paired-bootstrap deltas:

* **target**: every target slice (e.g. ``night``) improves by >= ``min_target_gain`` mAP@50:95, and
  (``require_ci``) the 95% CI lower bound of that gain is above zero: a gain we can trust.
* **no regression overall**: overall Delta >= -``max_overall_drop``.
* **no slice regression**: every slice with enough images (``gated``) has Delta >= -``max_slice_drop``.

The decision is pure (dict in, dict out), so CI can re-run it on committed evaluation artifacts.
"""

from __future__ import annotations

from failforge.config import GateConfig


def decide(base_eval: dict, cand_eval: dict, deltas: dict, cfg: GateConfig) -> dict:
    checks = []

    def add(name: str, value, threshold, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "value": value, "threshold": threshold, "passed": bool(passed), "detail": detail})

    for t in cfg.target_slices:
        d = deltas.get(t)
        if d is None:
            add(f"target:{t}", None, cfg.min_target_gain, False, "slice missing from the golden set")
            continue
        add(f"target:{t} gain", d["delta"], cfg.min_target_gain, d["delta"] >= cfg.min_target_gain,
            f"Delta mAP@50:95 = {d['delta']:+.4f} (95% CI {d['ci_low']:+.4f} .. {d['ci_high']:+.4f})")
        if cfg.require_ci:
            add(f"target:{t} CI > 0", d["ci_low"], 0.0, d["ci_low"] > 0, "bootstrap 95% CI lower bound")
    ov = deltas.get("overall")
    if ov is not None:
        add("overall no-regression", ov["delta"], -cfg.max_overall_drop, ov["delta"] >= -cfg.max_overall_drop,
            f"Delta overall = {ov['delta']:+.4f}")
    for name, s in cand_eval.get("slices", {}).items():
        if not s.get("gated") or name in cfg.target_slices:
            continue
        b = base_eval.get("slices", {}).get(name)
        if b is None or b.get("map") is None or s.get("map") is None:
            continue
        d = s["map"] - b["map"]
        add(f"slice:{name} no-regression", d, -cfg.max_slice_drop, d >= -cfg.max_slice_drop)
    promote = all(c["passed"] for c in checks) and bool(checks)
    return {"promote": promote, "checks": checks,
            "thresholds": {"min_target_gain": cfg.min_target_gain, "max_overall_drop": cfg.max_overall_drop,
                           "max_slice_drop": cfg.max_slice_drop, "require_ci": cfg.require_ci,
                           "target_slices": cfg.target_slices}}
