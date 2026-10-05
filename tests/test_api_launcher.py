import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _fake_run(home: Path, rid: str = "20260101-000000-quick") -> Path:
    d = home / "runs" / rid
    (d / "analyze" / "thumbs").mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"run_id": rid, "state": "done", "stage": None, "stages": [], "profile": "quick"}))
    (d / "config.json").write_text(json.dumps({"profile": "quick", "gate": {"target_slices": ["night"]}}))
    (d / "analyze" / "analysis.json").write_text(json.dumps({"points": [], "clusters": [], "modes": []}))
    (d / "analyze" / "thumbs" / "c0_0.jpg").write_bytes(b"\xff\xd8\xff")
    return d


@pytest.fixture
def client(workspace):
    from fastapi.testclient import TestClient

    from failforge.app.server import create_app

    return TestClient(create_app())


def test_api_health_overview_runs(client, workspace):
    _fake_run(workspace)
    assert client.get("/api/health").json()["ok"]
    o = client.get("/api/overview").json()
    assert set(o["profiles"]) == {"quick", "full", "smoke"} and not o["job"]["running"]
    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == "20260101-000000-quick" and runs[0]["state"] == "done"
    r = client.get("/api/runs/20260101-000000-quick").json()
    assert r["summary"]["profile"] == "quick" and r["gallery"] == []
    assert client.get("/api/runs/20260101-000000-quick/analysis").status_code == 200
    assert client.get("/api/runs/20260101-000000-quick/file/analyze/thumbs/c0_0.jpg").status_code == 200
    assert client.get("/").status_code == 200 and "FailForge" in client.get("/").text


def test_api_rejects_traversal_and_unknown(client, workspace):
    _fake_run(workspace)
    assert client.get("/api/runs/..%2F..%2Fetc").status_code in (400, 404)
    assert client.get("/api/runs/20260101-000000-quick/file/../../config.json").status_code == 404
    assert client.get("/api/runs/nope").status_code == 404
    assert client.post("/api/loop", json={"profile": "huge"}).status_code == 400


def test_api_stop_writes_stop_file(client, workspace):
    d = _fake_run(workspace)
    assert client.post(f"/api/runs/{d.name}/stop").json()["ok"]
    assert (d / "STOP").is_file()


def _launcher():
    spec = importlib.util.spec_from_file_location("ff_run", REPO / "run.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ff_run"] = m
    sys.dont_write_bytecode = True  # keep the repository root clean
    spec.loader.exec_module(m)
    return m


def test_launcher_plan_and_options():
    r = _launcher()
    keys = [s["key"] for s in r.plan("gpu")]
    assert keys == ["venv", "pip", "torch", "ort-swap", "app", "data", "prepare", "models", "verify"]
    gpu = next(s for s in r.plan("gpu") if s["key"] == "torch")["cmd"]
    cpu = next(s for s in r.plan("cpu") if s["key"] == "torch")["cmd"]
    assert gpu[-1].endswith("torch-gpu.txt") and cpu[-1].endswith("torch-cpu.txt")
    assert r.project_fingerprint() == r.project_fingerprint()
    assert r._take(["--home", "D:/x", "--cpu"], "--home") == ("D:/x", ["--cpu"])
    assert r.main(["--bogus"]) == 2
    assert r.main(["--loop", "huge"]) == 2


def test_launcher_uses_only_stdlib():
    import ast

    tree = ast.parse((REPO / "run.py").read_text(encoding="utf-8"))
    mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods
