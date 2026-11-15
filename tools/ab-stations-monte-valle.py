#!/usr/bin/env python3
"""A/B: add monte/valle stations to Cento MLP and compare flood metrics.

Strategies from prior gates (valle gate, lag-align, M9, WP1):
  - baseline = prod recipe: no chiusa, no valle acceptors, poly-points 6, flat window
  - skip LightGBM; gate on validateEvents + holdoutEvents skill@120
  - Bagnetto Reno is a direct Cento dependency but missing from prod UPSTREAMS
  - Dosso-only valle (nearest acceptor) instead of the full 4-acceptor set that failed
  - far monte / Samoggia tributaries as separate ablations
  - no lag-align, no flood-only, no quantile (already rejected for this gate)

Usage:
  uv run python tools/ab-stations-monte-valle.py
  uv run python tools/ab-stations-monte-valle.py --only baseline,bagnetto
  uv run python tools/ab-stations-monte-valle.py --summarize-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / "tools" / "fit-ml-cento.py"
OUT_JSON = ROOT / "models" / "ab-stations-monte-valle.json"
OUT_MD = ROOT / "models" / "ab-stations-monte-valle.md"

# Common flags: match production gate recipe.
COMMON = [
    "--no-chiusa",
    "--skip-lgbm",
    "--poly-points",
    "6",
    "--horizons",
    "1",
    "2",
    "3",
    "4",
    "-q",
]

VARIANTS: dict[str, dict] = {
    "baseline": {
        "label": "prod-like (4 monte, no valle)",
        "extra": [],
        "acceptors": None,  # --no-acceptors
        "hypothesis": "Reference matching cento-dnn / transfer-coeffs recipe",
    },
    "bagnetto": {
        "label": "+Bagnetto Reno (dep diretta)",
        "extra": ["Bagnetto Reno"],
        "acceptors": None,
        "hypothesis": "Nearest Reno gauge upstream of Cento fills a hole in the feature set",
    },
    "far_monte": {
        "label": "+Bagnetto + Vergato DX + Sasso Marconi",
        "extra": ["Bagnetto Reno", "Vergato DX", "Sasso Marconi"],
        "acceptors": None,
        "hypothesis": "Farther monte adds early timing for the rising limb",
    },
    "tributaries": {
        "label": "+Bagnetto + Lavino + Paltrone",
        "extra": ["Bagnetto Reno", "Lavino di Sotto", "Paltrone Samoggia"],
        "acceptors": None,
        "hypothesis": "Forcelli feeders improve Samoggia lateral timing",
    },
    "dosso": {
        "label": "Dosso-only valle (no extra monte)",
        "extra": [],
        "acceptors": ["Dosso"],
        "hypothesis": "Nearest valle backwater only (WP2.9 / H11 light test)",
    },
    "bagnetto_dosso": {
        "label": "+Bagnetto + Dosso",
        "extra": ["Bagnetto Reno"],
        "acceptors": ["Dosso"],
        "hypothesis": "Combine nearest monte hole-fill with nearest valle",
    },
    "bagnetto_dosso_chain": {
        "label": "+Bagnetto + Dosso + chain Δ",
        "extra": ["Bagnetto Reno"],
        "acceptors": ["Dosso"],
        "feature_set": "chain",
        "hypothesis": "Explicit Cento−Dosso slope feature for backwater",
    },
}


def build_cmd(name: str, cfg: dict) -> list[str]:
    tag_hint = name.replace("_", "-")
    cmd = [
        sys.executable,
        str(FIT),
        *COMMON,
        "--out",
        str(ROOT / "models" / f"ml-cento-compare-ab-{tag_hint}.json"),
        "--export-dnn",
        str(ROOT / "models" / f"cento-dnn-ab-{tag_hint}.json"),
    ]
    if cfg.get("feature_set"):
        cmd.extend(["--feature-set", cfg["feature_set"]])
    if cfg["acceptors"] is None:
        cmd.append("--no-acceptors")
    else:
        cmd.append("--acceptors")
        cmd.extend(cfg["acceptors"])
    if cfg["extra"]:
        cmd.append("--extra-upstreams")
        cmd.extend(cfg["extra"])
    return cmd


def extract_h120(compare_path: Path) -> dict | None:
    if not compare_path.exists():
        return None
    doc = json.loads(compare_path.read_text(encoding="utf-8"))
    block = (doc.get("horizons") or {}).get("120") or {}
    dnn = block.get("dnn") or {}
    return {
        "upstreams": doc.get("upstreams"),
        "acceptors": doc.get("acceptors"),
        "stations": doc.get("stations"),
        "featureSet": doc.get("featureSet"),
        "nFeatures": block.get("nFeatures"),
        "validateEvents": dnn.get("validateEvents"),
        "holdoutEvents": dnn.get("holdoutEvents"),
        "validate": dnn.get("validate"),
        "holdout": dnn.get("holdout"),
        "comparePath": str(compare_path.relative_to(ROOT)),
        "dnnPath": str(
            (ROOT / "models" / compare_path.name.replace("ml-cento-compare-", "cento-dnn-")).relative_to(ROOT)
        ),
    }


def skill(block: dict | None, key: str = "skill") -> float | None:
    if not block:
        return None
    v = block.get(key)
    return float(v) if v is not None else None


def decide(rows: list[dict]) -> dict:
    baseline = next((r for r in rows if r["id"] == "baseline" and r.get("metrics")), None)
    if not baseline:
        return {"promote": None, "reason": "baseline missing"}
    b_val = skill(baseline["metrics"].get("validateEvents"))
    b_hold = skill(baseline["metrics"].get("holdoutEvents"))
    best = None
    for r in rows:
        if r["id"] == "baseline" or not r.get("metrics"):
            continue
        v = skill(r["metrics"].get("validateEvents"))
        h = skill(r["metrics"].get("holdoutEvents"))
        if v is None or h is None or b_val is None or b_hold is None:
            continue
        # Gate: must not regress validate OR holdout event skill; prefer holdout.
        if v + 1e-4 >= b_val and h + 1e-4 >= b_hold:
            gain = (h - b_hold) + 0.25 * (v - b_val)
            if best is None or gain > best["gain"]:
                best = {"id": r["id"], "gain": gain, "val": v, "hold": h}
    if not best:
        return {
            "promote": None,
            "reason": (
                f"Nessuna variante batte baseline su validateEvents e holdoutEvents "
                f"(base val={b_val}, hold={b_hold}). Soft-adopt resta off."
            ),
        }
    return {
        "promote": best["id"],
        "gain": best["gain"],
        "val": best["val"],
        "hold": best["hold"],
        "baselineVal": b_val,
        "baselineHold": b_hold,
        "reason": (
            f"{best['id']} migliora holdoutEvents skill@120 "
            f"({b_hold:.4f}→{best['hold']:.4f}) senza regressione validate."
        ),
    }


def write_md(payload: dict, path: Path) -> None:
    lines = [
        "# A/B — stazioni monte/valle su Cento MLP",
        "",
        f"**Data:** {payload['comparedAt'][:10]}  ",
        f"**Gate:** validateEvents + holdoutEvents skill@120 (stesso split di `fit-ml-cento`)  ",
        "**Recipe comune:** `--no-chiusa --skip-lgbm --poly-points 6`, horizons 30…120′, flat window, no lag-align / flood-only / quantile.",
        "",
        "## Ipotesi",
        "",
        "| ID | Variante | Ipotesi |",
        "|---|---|---|",
    ]
    for r in payload["variants"]:
        lines.append(f"| `{r['id']}` | {r['label']} | {r['hypothesis']} |")
    lines += [
        "",
        "## Risultati @ 120′",
        "",
        "| ID | nFeat | valEvt skill | valEvt RMSE | holdEvt skill | holdEvt RMSE | upstreams extra / acceptors |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload["variants"]:
        m = r.get("metrics") or {}
        ve, he = m.get("validateEvents") or {}, m.get("holdoutEvents") or {}
        ups = m.get("upstreams") or []
        base_ups = {
            "Forcelli",
            "Sostegno Reno",
            "Bonconvento",
            "Casalecchio tiro a volo",
        }
        extra = [u for u in ups if u not in base_ups]
        acc = m.get("acceptors") or []
        note = f"extra={extra or '—'}; acc={acc or '—'}"
        lines.append(
            f"| `{r['id']}` | {m.get('nFeatures') or '—'} | "
            f"{ve.get('skill', '—') if ve else '—'} | {ve.get('rmse', '—') if ve else '—'} | "
            f"{he.get('skill', '—') if he else '—'} | {he.get('rmse', '—') if he else '—'} | {note} |"
        )
    d = payload["decision"]
    lines += [
        "",
        "## Decisione",
        "",
        f"- **promote:** `{d.get('promote')}`",
        f"- **motivo:** {d.get('reason')}",
        "",
        "Soft-adopt / patch `transfer-coeffs` solo se promote ≠ null e un protocol backtest conferma rising/phase.",
        "",
        "## File",
        "",
    ]
    for r in payload["variants"]:
        m = r.get("metrics") or {}
        if m.get("comparePath"):
            lines.append(f"- `{r['id']}`: `{m['comparePath']}` → `{m.get('dnnPath')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="Comma-separated variant ids")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    ap.add_argument("--md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    selected = list(VARIANTS)
    if args.only:
        selected = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = [x for x in selected if x not in VARIANTS]
        if unknown:
            raise SystemExit(f"Unknown variants: {unknown}")

    rows: list[dict] = []
    for name in selected:
        cfg = VARIANTS[name]
        compare = ROOT / "models" / f"ml-cento-compare-ab-{name.replace('_', '-')}.json"
        row = {
            "id": name,
            "label": cfg["label"],
            "hypothesis": cfg["hypothesis"],
            "cmd": build_cmd(name, cfg),
        }
        if not args.summarize_only:
            print(f"\n=== {name}: {cfg['label']} ===", flush=True)
            print(" ".join(row["cmd"]), flush=True)
            t0 = time.perf_counter()
            proc = subprocess.run(row["cmd"], cwd=ROOT)
            row["elapsedSec"] = round(time.perf_counter() - t0, 1)
            row["returncode"] = proc.returncode
            if proc.returncode != 0:
                print(f"! fit failed for {name} rc={proc.returncode}", flush=True)
        metrics = extract_h120(compare)
        # Fix dnn path to the explicit export name
        if metrics:
            metrics["dnnPath"] = f"models/cento-dnn-ab-{name.replace('_', '-')}.json"
            metrics["comparePath"] = f"models/ml-cento-compare-ab-{name.replace('_', '-')}.json"
        row["metrics"] = metrics
        rows.append(row)
        if metrics and metrics.get("validateEvents"):
            ve, he = metrics["validateEvents"], metrics["holdoutEvents"]
            print(
                f"→ {name}: valEvt skill={ve.get('skill')} rmse={ve.get('rmse')} | "
                f"holdEvt skill={he.get('skill')} rmse={he.get('rmse')}",
                flush=True,
            )

    decision = decide(rows)
    payload = {
        "version": 1,
        "kind": "ab_stations_monte_valle",
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "comparedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commonFlags": COMMON,
        "variants": rows,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, args.md)
    print(f"\nWrote {args.json}")
    print(f"Wrote {args.md}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()
