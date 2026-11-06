#!/usr/bin/env python3
"""Plots of method 2: mIoU against T4 latency, per (context model + glimpse model) pair.

    python -m scripts.method2_plots --sweep experiments/method2/sweep_train.jsonl \\
        --val results/method2/val.json --out results/method2

``train_<pair>.png``: every rule of the train sweep (uncertainty rules, the
prior, random), the pixel oracle per geometry (most correct pixels; not an mIoU ceiling), the
tiling bound (every cell glimpsed: drawn at the right edge, it has no T4 point),
the glimpse model's full-frame curve, and the finalists. ``val_<pair>.png``:
the finalists on val with their clip-bootstrap 95% intervals, against the
full-frame val curve, with the oracle of each finalist's geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KIND_STYLE = {"uncertainty": dict(color="#4C78A8", s=10, alpha=0.35, label="uncertainty rules"),
              "prior": dict(color="#72B7B2", s=14, alpha=0.6, marker="s", label="fixed prior"),
              "random": dict(color="#B0B0B0", s=10, alpha=0.6, marker="x", label="random (mean of 5 seeds)"),
              "oracle": dict(color="#E45756", s=14, alpha=0.6, marker="^", label="oracle: most correct pixels (labels)")}


def _curve(ax, fc: Dict[str, Any], label: str) -> None:
    pts = sorted(zip(fc["latency_ms"], fc["miou"], fc["scales"]))
    ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], "k-o", ms=4, lw=1.5, label=label, zorder=5)
    for lat, miou, s in pts:
        ax.annotate(f"{s:g}", (lat, 100 * miou), textcoords="offset points", xytext=(4, -10), fontsize=7)


def plot_train(rows: List[Dict[str, Any]], finalists: List[Dict[str, Any]], out: Path) -> List[Path]:
    paths = []
    for pair in sorted({r["method"]["pair"] for r in rows}):
        sub = [r for r in rows if r["method"]["pair"] == pair]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for kind, style in KIND_STYLE.items():
            pts = [(r["cost"]["t4_latency"]["median_ms"], 100 * r["metrics"]["miou"]) for r in sub
                   if r["method"]["rule"]["kind"] == kind and r["cost"]["t4_latency"]]
            if pts:
                ax.scatter(*zip(*pts), **style)
        tiling = [100 * r["metrics"]["miou"] for r in sub if r["method"]["rule"]["kind"] == "all"]
        if tiling:
            ax.axhline(max(tiling), color="#E45756", ls=":", lw=1, label=f"tiling bound (best {max(tiling):.1f})")
        _curve(ax, sub[0]["cost"]["fullframe_curve"], f"full frame ({sub[0]['method']['glimpse_model']})")
        fin = [(f["latency_ms"], 100 * f["miou"]) for f in finalists if f["method"]["pair"] == pair]
        if fin:
            ax.scatter(*zip(*fin), facecolors="none", edgecolors="#F58518", s=70, lw=1.5, label="finalists", zorder=6)
        ax.set(xlabel="predicted T4 latency, median (ms, CUDA graphs, fp16, batch 1)", ylabel="mIoU (%)",
               title=f"{pair}: VPS train")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        path = out / f"train_{pair}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_val(report: Dict[str, Any], out: Path) -> List[Path]:
    paths = []
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for f in report["finalists"]:
        by_pair.setdefault(f["method"]["pair"], []).append(f)
    for pair, fins in sorted(by_pair.items()):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _curve(ax, report["fullframe"][fins[0]["method"]["glimpse_model"]],
               f"full frame ({fins[0]['method']['glimpse_model']})")
        for f in fins:
            lat, miou = f["t4_latency"]["median_ms"], 100 * f["miou"]
            lo, hi = (100 * v if v is not None else miou for v in f["bootstrap"]["miou"])
            ax.errorbar(lat, miou, yerr=[[miou - lo], [hi - miou]], fmt="o", color="#F58518", capsize=3, zorder=6)
            ax.scatter(lat, 100 * f["references"]["oracle"], marker="^", color="#E45756", s=18, alpha=0.7)
        ax.scatter([], [], marker="^", color="#E45756", s=18, label="pixel oracle at the same geometry and k")
        ax.errorbar([], [], yerr=[], fmt="o", color="#F58518", label="finalists (95% clip bootstrap)")
        ax.set(xlabel="predicted T4 latency, median (ms)", ylabel="mIoU (%)", title=f"{pair}: VPS val")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        path = out / f"val_{pair}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="experiments/method2/sweep_train.jsonl")
    p.add_argument("--finalists", default="results/method2/finalists.json")
    p.add_argument("--val", default=None, help="results/method2/val.json (skip the val plots if absent).")
    p.add_argument("--out", default="results/method2")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.sweep).read_text().splitlines() if line.strip()]
    fin_path = Path(args.finalists)
    finalists = json.loads(fin_path.read_text())["finalists"] if fin_path.is_file() else []
    written = plot_train(rows, finalists, out)
    if args.val and Path(args.val).is_file():
        written += plot_val(json.loads(Path(args.val).read_text()), out)
    for path in written:
        print(f"[method2_plots] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
