"""A minimal, file-based model registry: versions, the current champion, lineage.

    registry/
      models/v1.pt, v2.pt ...
      champion.json     the model production should run, with its golden-set scores
      history.jsonl     every registration / promotion / rejection, append-only
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from failforge import paths


class Registry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or paths.registry_dir()
        (self.root / "models").mkdir(parents=True, exist_ok=True)

    @property
    def champion_file(self) -> Path:
        return self.root / "champion.json"

    def champion(self) -> dict | None:
        if not self.champion_file.is_file():
            return None
        c = json.loads(self.champion_file.read_text(encoding="utf-8"))
        return c if Path(c["weights"]).is_file() else None

    def _next_version(self) -> int:
        vs = [int(p.stem[1:]) for p in (self.root / "models").glob("v*.pt") if p.stem[1:].isdigit()]
        return max(vs, default=0) + 1

    def _log(self, event: dict) -> None:
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}) + "\n")

    def register(self, weights: Path, *, run_id: str, arm: str, golden: dict, parent: int | None,
                 promote: bool, note: str = "") -> dict:
        v = self._next_version()
        dst = self.root / "models" / f"v{v}.pt"
        shutil.copy2(weights, dst)
        entry = {"version": v, "weights": str(dst.resolve()), "run_id": run_id, "arm": arm, "parent": parent,
                 "golden": golden, "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note}
        if promote:
            self.champion_file.write_text(json.dumps(entry, indent=1), encoding="utf-8")
        self._log({"event": "promoted" if promote else "registered", **entry})
        return entry

    def reject(self, *, run_id: str, arm: str, reason: str) -> None:
        self._log({"event": "rejected", "run_id": run_id, "arm": arm, "reason": reason})

    def history(self) -> list[dict]:
        p = self.root / "history.jsonl"
        if not p.is_file():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
