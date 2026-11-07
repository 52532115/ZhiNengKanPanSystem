"""版本化数据缓存：每份数据都带来源、抓取时间、行数与校验和。

为什么不用裸文件
----------------
回测结论必须可复现，而"昨天跑出来 IC 是 +0.013、今天变成 +0.021"这种漂移，
八成不是模型变了，是数据被静默更新或补拉了。这里的规矩：

1. 每次抓取生成一个**版本号**（``pit-YYYYMMDDTHHMM``），写进 ``manifest.json``；
2. 每个数据文件登记来源、抓取时间、行数、首末日期、sha256；
3. 回测入口只认版本号，不认"当前缓存"，所以同一版本永远能重放；
4. ``verify()`` 能发现文件被改动或缺失，报告里会打印版本与校验结果。
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT / ".cache" / "research"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


class Dataset:
    """一个版本目录的读写句柄。"""

    def __init__(self, version: str, *, root: Path | None = None) -> None:
        self.version = version
        self.root = (root or CACHE_ROOT) / version
        self.manifest_path = self.root / "manifest.json"

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        *,
        version: str | None = None,
        params: dict[str, Any] | None = None,
        root: Path | None = None,
    ) -> "Dataset":
        version = version or datetime.now().strftime("pit-%Y%m%dT%H%M")
        dataset = cls(version, root=root)
        dataset.root.mkdir(parents=True, exist_ok=True)
        (dataset.root / "bars").mkdir(exist_ok=True)
        dataset.write_manifest(
            {
                "version": version,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "params": params or {},
                "sources": {},
                "files": {},
            }
        )
        return dataset

    @classmethod
    def open(cls, version: str, *, root: Path | None = None) -> "Dataset":
        dataset = cls(version, root=root)
        if not dataset.manifest_path.is_file():
            raise FileNotFoundError(f"数据版本不存在：{dataset.root}")
        return dataset

    def write_manifest(self, payload: dict[str, Any]) -> None:
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def update_manifest(self, **fields: Any) -> None:
        payload = self.manifest()
        payload.update(fields)
        self.write_manifest(payload)

    def register_source(self, name: str, **info: Any) -> None:
        payload = self.manifest()
        payload.setdefault("sources", {})[name] = {
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            **info,
        }
        self.write_manifest(payload)

    # ------------------------------------------------------------------ #
    # 表数据
    # ------------------------------------------------------------------ #
    def write_table(self, name: str, payload: Any, *, source: str, rows: int | None = None) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self._register_file(name, path, source=source, rows=rows)
        return path

    def read_table(self, name: str) -> Any:
        return json.loads((self.root / f"{name}.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # 逐标的日线（gzip CSV，避免上万行 JSON 的体积）
    # ------------------------------------------------------------------ #
    BAR_COLUMNS = (
        "date", "open", "high", "low", "close", "preclose",
        "volume", "amount", "turn", "tradestatus", "isST",
    )

    def bar_path(self, code: str) -> Path:
        return self.root / "bars" / f"{code.replace('.', '_')}.csv.gz"

    def write_bars(self, code: str, rows: Iterable[tuple[str, ...]]) -> int:
        path = self.bar_path(code)
        count = 0
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            handle.write(",".join(self.BAR_COLUMNS) + "\n")
            for row in rows:
                handle.write(",".join(row) + "\n")
                count += 1
        return count

    def read_bars(self, code: str) -> list[dict[str, Any]]:
        path = self.bar_path(code)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                cells = line.split(",")
                rows.append(dict(zip(header, cells)))
        return rows

    def has_bars(self, code: str) -> bool:
        return self.bar_path(code).is_file()

    # ------------------------------------------------------------------ #
    # 校验
    # ------------------------------------------------------------------ #
    def _register_file(self, name: str, path: Path, *, source: str, rows: int | None) -> None:
        payload = self.manifest()
        payload.setdefault("files", {})[name] = {
            "path": path.name,
            "rows": rows,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "source": source,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.write_manifest(payload)

    def register_bars(self, code: str, count: int) -> None:
        payload = self.manifest()
        entry = payload.setdefault("bar_index", {}).setdefault(code, {})
        entry.update({"rows": count})
        self.write_manifest(payload)

    def finalize_bars(self) -> None:
        """一次性登记所有日线文件的校验和，避免每只股票都重写 manifest。"""
        payload = self.manifest()
        index = {}
        for path in sorted((self.root / "bars").glob("*.csv.gz")):
            code = path.name.replace("_", ".").replace(".csv.gz", "")
            prior = payload.get("bar_index", {}).get(code, {})
            index[code] = {
                "rows": prior.get("rows"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        payload["bar_index"] = index
        payload["bars_total"] = len(index)
        self.write_manifest(payload)

    def verify(self) -> dict[str, Any]:
        """校验和 + 缺失检查，报告开头会打印。"""
        payload = self.manifest()
        missing, changed = [], []
        for name, info in payload.get("files", {}).items():
            path = self.root / info["path"]
            if not path.is_file():
                missing.append(name)
            elif _sha256(path) != info["sha256"]:
                changed.append(name)
        bars = payload.get("bar_index", {})
        bar_missing = [code for code, info in bars.items() if not self.bar_path(code).is_file()]
        return {
            "version": self.version,
            "created_at": payload.get("created_at"),
            "tables": len(payload.get("files", {})),
            "bars": payload.get("bars_total") or len(bars),
            "missing": missing,
            "changed": changed,
            "bars_missing": bar_missing,
            "ok": not missing and not changed and not bar_missing,
        }


def latest_version(*, root: Path | None = None) -> str | None:
    base = root or CACHE_ROOT
    if not base.is_dir():
        return None
    candidates = sorted(p.name for p in base.iterdir() if (p / "manifest.json").is_file())
    return candidates[-1] if candidates else None
