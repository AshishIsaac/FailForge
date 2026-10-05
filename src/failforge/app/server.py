"""FailForge dashboard: a local web app over the workspace.

* browse runs (the committed reference run ships with the repository, so there is something to see
  before the first local loop finishes),
* start / stop / resume a loop: it runs as a separate ``python -m failforge loop`` process, so the
  dashboard stays responsive and a browser refresh (or closing it) never kills a training run,
* watch it live: the pipeline writes ``status.json``; the page polls it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from failforge import __version__, paths
from failforge.loop.pipeline import STAGES, new_run_id

log = logging.getLogger("failforge.app")
STATIC = Path(__file__).parent / "static"
PROFILES = {
    "quick": {"title": "Quick", "detail": "1,500 day images, 8+4 epochs, 150 ControlNet images", "gpu": "45-75 min", "cpu": "many hours"},
    "full": {"title": "Full (reference)", "detail": "all day images, 30+12 epochs, 1,000 ControlNet images", "gpu": "4-6 h", "cpu": "not practical"},
    "smoke": {"title": "Smoke test", "detail": "40 images, 1 epoch, classical synthesis: checks the wiring", "gpu": "3-5 min", "cpu": "5-10 min"},
}


class StartRequest(BaseModel):
    profile: str = "quick"
    run_id: str | None = None  # resume this run


class JobManager:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.run_id: str | None = None
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, profile: str, run_id: str | None) -> str:
        with self.lock:
            if self.running():
                raise HTTPException(409, f"a loop is already running ({self.run_id})")
            if profile not in PROFILES:
                raise HTTPException(400, f"unknown profile {profile}")
            rid = run_id or new_run_id(profile)
            d = paths.runs_dir() / rid
            d.mkdir(parents=True, exist_ok=True)
            (d / "STOP").unlink(missing_ok=True)
            con = (d / "console.log").open("ab")
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self.proc = subprocess.Popen([sys.executable, "-m", "failforge", "loop", "-c", profile, "--run-id", rid],
                                         cwd=paths.REPO, stdout=con, stderr=subprocess.STDOUT, env=env, creationflags=flags,
                                         start_new_session=os.name != "nt")  # Ctrl+C on the dashboard spares the loop
            self.run_id = rid
            log.info("started loop %s (pid %d)", rid, self.proc.pid)
            return rid

    def stop(self, run_id: str) -> None:
        d = paths.runs_dir() / run_id
        if d.is_dir():
            (d / "STOP").write_text("stop requested from the dashboard")

    def info(self) -> dict:
        return {"running": self.running(), "run_id": self.run_id,
                "exit_code": None if self.proc is None or self.running() else self.proc.returncode}


def _run_dir(rid: str) -> Path:
    if rid == "reference":
        d = paths.reference_dir()
    else:
        if "/" in rid or "\\" in rid or rid.startswith("."):
            raise HTTPException(400, "bad run id")
        d = paths.runs_dir() / rid
    if not d.is_dir():
        raise HTTPException(404, f"run {rid} not found")
    return d


def _j(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, ValueError):
        return None


def _summary(rid: str, d: Path) -> dict:
    st = _j(d / "status.json") or {}
    cfg = _j(d / "config.json") or {}
    gate = _j(d / "gate" / "gate.json")
    state = st.get("state", "unknown")
    if state == "running" and rid != "reference" and time.time() - (d / "status.json").stat().st_mtime > 600:
        pid = st.get("pid")
        if pid and not _pid_alive(pid):
            state = "interrupted"
    return {"id": rid, "profile": cfg.get("profile", st.get("profile")), "state": state, "updated": st.get("updated"),
            "stage": st.get("stage"), "verdict": gate.get("verdict") if gate else None,
            "promote": gate["decision"]["promote"] if gate else None, "reference": rid == "reference"}


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        return True


def create_app() -> FastAPI:
    app = FastAPI(title="FailForge", version=__version__)
    jobs = JobManager()
    app.state.jobs = jobs

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/api/overview")
    def overview() -> dict:
        from failforge.loop.registry import Registry

        gpu = None
        try:
            import torch

            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                gpu = f"{p.name} ({p.total_memory / 2**30:.0f} GB)"
        except ImportError:
            pass
        lock = _j(paths.manifests_dir() / "golden.lock.json")
        champs = {}
        for prof in PROFILES:
            c = Registry(paths.registry_dir() / prof).champion() if (paths.registry_dir() / prof).is_dir() else None
            if c:
                champs[prof] = {"version": c["version"], "arm": c["arm"], "run_id": c["run_id"], "golden": c["golden"]}
        return {"version": __version__, "workspace": str(paths.home()), "gpu": gpu,
                "data_ready": lock is not None and (paths.manifests_dir() / "train_day.jsonl").is_file(),
                "golden": {"n": lock["n"], "created": lock["created"], "digest": lock["digest"][:12]} if lock else None,
                "champions": champs, "profiles": PROFILES, "job": jobs.info(),
                "stages": [{"name": n, "title": t} for n, t, _ in STAGES]}

    @app.get("/api/runs")
    def runs() -> list[dict]:
        out = []
        rd = paths.runs_dir()
        if rd.is_dir():
            for d in sorted((p for p in rd.iterdir() if p.is_dir() and (p / "status.json").is_file()),
                            key=lambda p: p.name, reverse=True):
                out.append(_summary(d.name, d))
        if paths.reference_dir().is_dir():
            out.append(_summary("reference", paths.reference_dir()))
        return out

    @app.get("/api/runs/{rid}")
    def run(rid: str) -> dict:
        d = _run_dir(rid)
        ev = d / "eval"
        res = {"summary": _summary(rid, d), "status": _j(d / "status.json"), "config": _j(d / "config.json"),
               "monitor": _j(d / "monitor" / "monitor.json"), "probe": _j(d / "mine" / "probe_eval.json"),
               "synthesis": _j(d / "synthesize" / "synthesis.json"), "gate": _j(d / "gate" / "gate.json"),
               "ablation": _j(ev / "ablation.json"), "deltas": _j(ev / "deltas.json"),
               "baseline": _j(ev / "baseline.json"), "has_report": (d / "report" / "report.html").is_file()}
        gal = d / "synthesize" / "synthetic" / "gallery"
        res["gallery"] = sorted(p.name for p in gal.glob("*.jpg")) if gal.is_dir() else []
        aug = d / "synthesize" / "augment" / "gallery"
        res["gallery_augment"] = sorted(p.name for p in aug.glob("*.jpg"))[:4] if aug.is_dir() else []
        return res

    @app.get("/api/runs/{rid}/analysis")
    def analysis(rid: str):
        a = _run_dir(rid) / "analyze" / "analysis.json"
        if not a.is_file():
            raise HTTPException(404, "no analysis yet")
        return FileResponse(a, media_type="application/json")

    @app.get("/api/runs/{rid}/log", response_class=PlainTextResponse)
    def run_log(rid: str, tail: int = 200) -> str:
        d = _run_dir(rid)
        for name in ("console.log", "run.log"):
            p = d / name
            if p.is_file():
                with p.open("rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(size - 200_000, 0))
                    lines = f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
                lines = [x for x in lines if x.strip()]
                return "\n".join(lines[-tail:])
        return ""

    @app.get("/api/runs/{rid}/file/{rel:path}")
    def run_file(rid: str, rel: str):
        d = _run_dir(rid).resolve()
        p = (d / rel).resolve()
        if d not in p.parents or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p)

    @app.post("/api/loop")
    def start(req: StartRequest) -> dict:
        return {"run_id": jobs.start(req.profile, req.run_id)}

    @app.post("/api/runs/{rid}/stop")
    def stop(rid: str) -> dict:
        _run_dir(rid)
        jobs.stop(rid)
        return {"ok": True}

    @app.get("/api/job")
    def job() -> dict:
        return jobs.info()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(FileNotFoundError)
    def _fnf(_, e: FileNotFoundError):
        return JSONResponse({"detail": str(e)}, status_code=404)

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, autostart: str | None = None) -> int:
    import socket

    import uvicorn

    for p in range(port, port + 20):  # first free port
        with socket.socket() as s:
            if s.connect_ex((host, p)) != 0:
                port = p
                break
    app = create_app()
    if autostart:
        app.state.jobs.start(autostart, None)
    url = f"http://{host}:{port}/"
    ready = os.environ.get("FAILFORGE_READY_FILE")

    def announce() -> None:
        for _ in range(100):
            time.sleep(0.1)
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    break
            except OSError:
                continue
        print(f"\nFailForge dashboard: {url}   (Ctrl+C to quit; a running loop keeps going in the background)\n", flush=True)
        if ready:
            Path(ready).write_text(url)
        if open_browser:
            webbrowser.open(url)

    threading.Thread(target=announce, daemon=True).start()
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(app, host=host, port=port, log_level="warning")
    print("dashboard closed.")
    return 0


if __name__ == "__main__":
    from failforge.cli import main

    sys.exit(main(["serve", *sys.argv[1:]]))
