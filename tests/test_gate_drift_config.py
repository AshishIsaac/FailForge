import numpy as np
import pytest

from failforge.analysis.drift import drift_report, mmd_test, psi
from failforge.config import ConfigError, GateConfig, load_config
from failforge.loop.gate import decide


def _eval(overall, **slices):
    return {"overall": {"map": overall}, "slices": {k: {"map": v, "gated": True} for k, v in slices.items()}}


def _d(delta, lo, hi):
    return {"delta": delta, "ci_low": lo, "ci_high": hi}


def test_gate_promotes_a_clear_win():
    base, cand = _eval(0.30, day=0.35, night=0.20), _eval(0.31, day=0.35, night=0.25)
    dec = decide(base, cand, {"night": _d(0.05, 0.03, 0.07), "overall": _d(0.01, 0.0, 0.02)}, GateConfig())
    assert dec["promote"], dec


@pytest.mark.parametrize("why,cand,deltas", [
    ("gain too small", _eval(0.30, day=0.35, night=0.21), {"night": _d(0.01, 0.0, 0.02), "overall": _d(0, 0, 0)}),
    ("CI crosses zero", _eval(0.30, day=0.35, night=0.23), {"night": _d(0.03, -0.01, 0.06), "overall": _d(0, 0, 0)}),
    ("overall regresses", _eval(0.29, day=0.35, night=0.25), {"night": _d(0.05, 0.03, 0.07), "overall": _d(-0.01, -0.02, 0)}),
    ("day slice regresses", _eval(0.30, day=0.30, night=0.25), {"night": _d(0.05, 0.03, 0.07), "overall": _d(0, 0, 0)}),
])
def test_gate_rejects(why, cand, deltas):
    base = _eval(0.30, day=0.35, night=0.20)
    assert not decide(base, cand, deltas, GateConfig())["promote"], why


def test_mmd_null_vs_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(80, 8)), rng.normal(size=(80, 8))
    assert mmd_test(a, b, permutations=100)["p_value"] > 0.05
    assert mmd_test(a, b + 1.0, permutations=100)["p_value"] < 0.02


def test_drift_report_and_psi():
    rng = np.random.default_rng(1)
    ref, cur = rng.normal(size=(60, 6)), rng.normal(size=(60, 6)) + 0.8
    r = drift_report(ref, cur, rng.uniform(0.4, 0.6, 60), rng.uniform(0.1, 0.3, 60), permutations=100)
    assert r["drift"] and r["domain_auc"] > 0.8 and r["psi_brightness"] > 1.0
    x = rng.normal(size=500)
    assert psi(x, x) < 0.01


def test_config_strict_profiles_and_overrides(tmp_path):
    q = load_config("quick")
    assert q.profile == "quick" and q.synthesis.count == 150 and q.gate.target_slices == ["night"]  # inherited
    assert load_config("smoke", ["synthesis.count=3"]).synthesis.count == 3
    bad = tmp_path / "bad.yaml"
    bad.write_text("synthesis:\n  cuont: 3\n")
    with pytest.raises(ConfigError, match="cuont"):
        load_config(bad)
    with pytest.raises(ConfigError):
        load_config(None, ["synthesis.backend=gan"])
    assert load_config("full").fingerprint("gate") != load_config("full", ["gate.min_target_gain=0.5"]).fingerprint("gate")
