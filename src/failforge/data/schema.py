"""Core data records and the manifest format.

A *manifest* is a JSON-lines file, one :class:`Sample` per line: image path, size, attributes
(``timeofday``, ``weather``, ``scene``, ``source``) and boxes in absolute pixel ``xyxy``. Every split,
the synthetic set and each arm's training set is a manifest, so the whole pipeline speaks one
format and every set can be validated, hashed and diffed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from failforge.config import CLASSES


@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    occluded: bool = False
    truncated: bool = False

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Sample:
    id: str
    image: str  # absolute path, or relative to the manifest's folder
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_dict(d: dict) -> Sample:
        return Sample(id=d["id"], image=d["image"], width=int(d["width"]), height=int(d["height"]),
                      boxes=[Box(**b) for b in d.get("boxes", [])], attrs=dict(d.get("attrs", {})))


class ManifestError(ValueError):
    pass


def write_manifest(path: Path, samples: Iterable[Sample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_json() + "\n")
            n += 1
    tmp.replace(path)
    return n


def iter_manifest(path: Path) -> Iterator[Sample]:
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Sample.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError) as e:
                raise ManifestError(f"{path}:{i}: {e}") from e


def read_manifest(path: Path) -> list[Sample]:
    return list(iter_manifest(path))


def resolve_image(sample: Sample, manifest_path: Path | None = None) -> Path:
    p = Path(sample.image)
    if not p.is_absolute() and manifest_path is not None:
        p = manifest_path.parent / p
    return p


def manifest_digest(samples: Iterable[Sample]) -> str:
    """Order-independent content hash of ids + labels (what the golden lock pins)."""
    h = hashlib.sha256()
    for line in sorted(json.dumps({"id": s.id, "boxes": [asdict(b) for b in s.boxes]}, sort_keys=True)
                       for s in samples):
        h.update(line.encode())
    return h.hexdigest()


def validate_samples(samples: list[Sample], *, check_files: bool = False, manifest_path: Path | None = None) -> list[str]:
    """Schema / sanity checks. Returns a list of problems (empty = valid)."""
    problems: list[str] = []
    seen: set[str] = set()
    for s in samples:
        if s.id in seen:
            problems.append(f"{s.id}: duplicate id")
        seen.add(s.id)
        if s.width <= 0 or s.height <= 0:
            problems.append(f"{s.id}: bad image size {s.width}x{s.height}")
        for j, b in enumerate(s.boxes):
            if not 0 <= b.cls < len(CLASSES):
                problems.append(f"{s.id} box {j}: class {b.cls} out of range")
            if not (b.x2 > b.x1 and b.y2 > b.y1):
                problems.append(f"{s.id} box {j}: degenerate {b.xyxy()}")
            if b.x1 < -1 or b.y1 < -1 or b.x2 > s.width + 1 or b.y2 > s.height + 1:
                problems.append(f"{s.id} box {j}: outside the image {b.xyxy()}")
        if check_files and not resolve_image(s, manifest_path).is_file():
            problems.append(f"{s.id}: image missing ({s.image})")
        if len(problems) > 200:
            problems.append("... (truncated)")
            break
    return problems
