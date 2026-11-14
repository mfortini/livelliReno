#!/usr/bin/env python3
"""Export slim Cento Q10/Q90(+conformal) sidecar for Fase 3 soft-merge.

Uso:
  uv run python tools/export-cento-bands.py \\
    --src models/cento-dnn-quantiles.json --out models/cento-dnn-bands.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "models" / "cento-dnn-quantiles.json")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "cento-dnn-bands.json")
    args = ap.parse_args()
    src = json.loads(args.src.read_text(encoding="utf-8"))
    out = {
        "version": 1,
        "kind": "window_delta_mlp_bands",
        "downstream": "Cento",
        "quantiles": src.get("quantiles") or [0.1, 0.5, 0.9],
        "conformalAlpha": src.get("conformalAlpha", 0.2),
        "note": "Fase 3: solo layersQ10/Q90 + conformalQhat; mediana resta da transfer-coeffs.",
        "source": str(args.src.relative_to(ROOT)) if args.src.is_absolute() else str(args.src),
        "horizons": {},
    }
    for key, block in (src.get("horizons") or {}).items():
        if not block.get("layersQ10") or not block.get("layersQ90"):
            continue
        out["horizons"][key] = {
            "layersQ10": block["layersQ10"],
            "layersQ90": block["layersQ90"],
            "conformalQhat": block.get("conformalQhat", 0),
            "coverageValidateConformal": block.get("coverageValidateConformal"),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} horizons={len(out['horizons'])} MB={args.out.stat().st_size / 1e6:.1f}")


if __name__ == "__main__":
    main()
