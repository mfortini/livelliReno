#!/usr/bin/env python3
"""Build models/skill-gates.json from Fase 0 eval-table (skill@120 CI).

Uso:
  uv run python tools/export-skill-gates.py \\
    --eval models/eval-table.json --out models/skill-gates.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=ROOT / "models" / "eval-table.json")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "skill-gates.json")
    ap.add_argument("--enter", type=float, default=0.10)
    ap.add_argument("--exit", type=float, default=0.05)
    args = ap.parse_args()
    table = json.loads(args.eval.read_text(encoding="utf-8"))
    stations = {}
    for name, mods in (table.get("stations") or {}).items():
        prod = (mods.get("prod") or {}).get("120") or {}
        ci = prod.get("skillEventPersistCI") or {}
        stations[name] = {
            "skill120": prod.get("skillEventPersist"),
            "skill120CI": {
                "lo": ci.get("lo"),
                "hi": ci.get("hi"),
                "nEvents": ci.get("nEvents") or prod.get("nEvents"),
            },
        }
    out = {
        "version": 1,
        "kind": "skill_gates",
        "disclaimer": "Sperimentale. CI da protocollo Fase 0 sul test congelato.",
        "testFrom": (table.get("from") or "2026-01-01T00:00:00.000Z"),
        "skillEnter": args.enter,
        "skillExit": args.exit,
        "stations": stations,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} stations={len(stations)}")


if __name__ == "__main__":
    main()
