#!/usr/bin/env python3
"""Diagnose a local copy of a video segmentation dataset before trusting it.

Written for the first contact with the real Cityscapes-VPS data, but
dataset-agnostic: it only uses the :class:`data.base.VideoSegDataset`
interface plus the adapter's optional diagnostic hooks, and every
dataset-specific choice comes from the config.

    export CITYSCAPES_VPS_ROOT=/path/to/cityscapes_vps
    python scripts/diagnose_real_data.py --config configs/diagnose_cityscapes_vps.yaml

    # dry run on the synthetic official-layout fixture:
    python scripts/diagnose_real_data.py --fixture

Per split it prints (and stores in diagnose_report.json):

* the on-disk layout (root / split / image / annotation folders);
* files found, how many match the filename regexes, annotations joined to a
  frame, orphan annotations (adapter hook);
* number of clips and frames, frames and labelled frames per clip, labelled
  positions inside clips;
* image (and annotation) resolution(s);
* for a sample of annotations: the unique raw ids and the class each decodes to;
* a histogram of the train classes plus ignore over the sampled annotations;
* pixel agreement with alternative annotation sources (config ``crosscheck``);

and saves a few overlay images (frame, frame + decoded labels, labels) so the
mapping can be checked by eye. Nothing here raises on a mismatch: problems
are listed at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # `python scripts/diagnose_real_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import REPO_ROOT, apply_overrides, ensure_fixture, ensure_src_on_path, expand_env, load_yaml  # noqa: E402

ensure_src_on_path()

import numpy as np  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "diagnose_cityscapes_vps.yaml"
DEFAULT_OUT = REPO_ROOT / "experiments" / "data_check" / "diagnose"


class Report:
    """Prints sections, isolates their failures and collects problems."""

    def __init__(self) -> None:
        self.problems: List[str] = []

    @contextlib.contextmanager
    def section(self, title: str) -> Iterator[None]:
        print(f"\n--- {title} ---")
        try:
            yield
        except Exception as exc:  # a broken section must not stop the others
            self.problem(f"{title}: {type(exc).__name__}: {exc}")
            print("    " + traceback.format_exc().strip().splitlines()[-1])

    def problem(self, message: str) -> None:
        print(f"  !! {message}")
        self.problems.append(message)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="Diagnose config (default: %(default)s).")
    p.add_argument("--root", default=None, help="Dataset root (overrides the config / environment variable).")
    p.add_argument("--fixture", action="store_true", help="Use the synthetic official-layout fixture.")
    p.add_argument("--splits", nargs="+", default=None, help="Splits to check (default: from the config).")
    p.add_argument("--sample", type=int, default=None, help="Annotations to sample per split.")
    p.add_argument("--overlays", type=int, default=None, help="Overlay images to save in total.")
    p.add_argument("--max-resolution-frames", type=int, default=0,
                   help="Frames whose size is read per split (0 = all; header-only reads when supported).")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config value.")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output folder for overlays + report (default: %(default)s).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _fmt_counter(counter: Counter, limit: int = 12) -> str:
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    text = ", ".join(f"{k}: {v}" for k, v in items[:limit])
    return text + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def _evenly(items: Sequence[Any], k: int) -> List[Any]:
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    idx = np.unique(np.linspace(0, len(items) - 1, k).round().astype(int))
    return [items[i] for i in idx]


def _labelled_frames(ds) -> List[Tuple[int, int]]:
    return [(ci, int(t)) for ci in range(len(ds)) for t in np.flatnonzero(ds.labeled_mask(ci))]


def compose_overlay(frame: np.ndarray, label: np.ndarray, palette: np.ndarray, ignore: int,
                    class_names: Sequence[str], title: str, alpha: float = 0.55):
    """Frame / frame+labels / labels stacked vertically, with a legend (PIL image)."""
    from PIL import Image, ImageDraw

    num_classes = len(class_names)
    valid = (label != ignore) & (label >= 0) & (label < num_classes)
    colour = np.zeros_like(frame)
    colour[valid] = palette[label[valid]]
    blend = frame.copy()
    blend[valid] = ((1 - alpha) * frame[valid] + alpha * colour[valid]).astype(np.uint8)
    height, width = label.shape
    target_w = int(min(max(width, 512), 1024))
    target_h = max(1, round(height * target_w / width))
    panels = [
        Image.fromarray(frame).resize((target_w, target_h), Image.BILINEAR),
        Image.fromarray(blend).resize((target_w, target_h), Image.NEAREST),
        Image.fromarray(colour).resize((target_w, target_h), Image.NEAREST),
    ]
    counts = Counter(label[valid].tolist())
    present = sorted(counts, key=lambda c: -counts[c])
    legend_rows = (len(present) + 1 + 2) // 3
    header, gap, row_h = 18, 4, 16
    canvas = Image.new("RGB", (target_w, header + 3 * (target_h + gap) + legend_rows * row_h + 8), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), title, fill=(255, 255, 255))
    for k, panel in enumerate(panels):
        canvas.paste(panel, (0, header + k * (target_h + gap)))
    y0 = header + 3 * (target_h + gap) + 4
    total = max(label.size, 1)
    entries = [(tuple(int(v) for v in palette[c]), f"{class_names[c]} {100 * counts[c] / total:.1f}%") for c in present]
    entries.append(((0, 0, 0), f"ignore {100 * (~valid).sum() / total:.1f}%"))
    col_w = target_w // 3
    for i, (rgb, text) in enumerate(entries):
        x, y = (i % 3) * col_w + 4, y0 + (i // 3) * row_h
        draw.rectangle([x, y + 2, x + 10, y + 12], fill=rgb, outline=(200, 200, 200))
        draw.text((x + 14, y), text, fill=(230, 230, 230))
    return canvas


# --------------------------------------------------------------------------- #
# One split
# --------------------------------------------------------------------------- #
def diagnose_split(split: str, spec: Dict[str, Any], cfg: Dict[str, Any], args: argparse.Namespace,
                   rep: Report, out_dir: Path, overlays_left: int) -> Tuple[Dict[str, Any], int]:
    from data.registry import build_dataset, resolve_dataset_class

    result: Dict[str, Any] = {"split": split}
    spec = dict(spec, split=split)
    print(f"\n================ split: {split} ================")

    with rep.section("on-disk layout"):
        layout = resolve_dataset_class(str(spec["type"])).layout_report(spec, base_dir=REPO_ROOT)
        result["layout"] = layout
        for key, info in layout.items():
            if isinstance(info, dict):
                state = "MISSING" if info.get("exists") is False else (
                    f"{info.get('num_entries')} entries: {info.get('entries')}" if info.get("exists") else "")
                print(f"  {key:15s} {info.get('path')}  {state}")
            else:
                print(f"  {key:15s} {info}")

    dataset = None
    with rep.section("build dataset"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset = build_dataset(spec, base_dir=REPO_ROOT)
        result["build_warnings"] = [str(w.message) for w in caught]
        for w in result["build_warnings"]:
            print(f"  warning: {w}")
        print(f"  {type(dataset).__name__}: {len(dataset)} clips")
    if dataset is None:
        rep.problem(f"[{split}] dataset could not be built (see above); skipping this split")
        return result, overlays_left

    with rep.section("indexing (adapter diagnostics)"):
        diag = dataset.diagnostics()
        result["diagnostics"] = diag
        index = diag.get("index", {})
        if index:
            print(f"  image files: {index.get('image_files')} in {index.get('image_dir')} (glob {index.get('image_glob')!r})")
            print(f"  filename regex {index.get('image_regex')!r}: {index.get('image_regex_matched')} match, "
                  f"{index.get('image_regex_unmatched')} do not  e.g. {index.get('image_regex_unmatched_examples')}")
            print(f"  annotation files: {index.get('annotation_files')} in {index.get('annotation_dir')} "
                  f"(exists: {index.get('annotation_dir_exists')})")
            if index.get("annotation_regex"):
                print(f"  annotation regex {index.get('annotation_regex')!r}: {index.get('annotation_regex_matched')} "
                      f"match, {index.get('annotation_regex_unmatched')} do not  "
                      f"e.g. {index.get('annotation_regex_unmatched_examples')}")
            print(f"  frames with an annotation: {index.get('frames_with_annotation')}; orphan annotations "
                  f"(no matching frame): {index.get('orphan_annotations')} e.g. {index.get('orphan_annotation_examples')}")
            if index.get("image_regex_unmatched"):
                rep.problem(f"[{split}] {index['image_regex_unmatched']} image files do not match the filename regex")
            if index.get("orphan_annotations"):
                rep.problem(f"[{split}] {index['orphan_annotations']} annotations match no frame")
            if index.get("annotation_duplicate_keys"):
                rep.problem(f"[{split}] {index['annotation_duplicate_keys']} annotations share a key")
            if index.get("annotation_clip_id_mismatches"):
                rep.problem(f"[{split}] {index['annotation_clip_id_mismatches']} frames: clip id differs from annotation")
            for msg in index.get("clip_split_warnings", []):
                rep.problem(f"[{split}] {msg}")
        for j in diag.get("side_car_jsons", []):
            print(f"  side-car {j}")
        if diag.get("label_mapping"):
            lm = {k: v for k, v in diag["label_mapping"].items() if k != "class_names"}
            print(f"  label mapping: {lm}")

    with rep.section("clips and frames"):
        n_frames = [dataset.num_frames(i) for i in range(len(dataset))]
        n_lab = [int(dataset.labeled_mask(i).sum()) for i in range(len(dataset))]
        positions = Counter(int(t) for i in range(len(dataset)) for t in np.flatnonzero(dataset.labeled_mask(i)))
        result["clips"] = {
            "num_clips": len(dataset), "num_frames": int(sum(n_frames)), "num_labeled_frames": int(sum(n_lab)),
            "frames_per_clip": dict(Counter(n_frames)), "labeled_per_clip": dict(Counter(n_lab)),
            "labeled_positions": dict(sorted(positions.items())),
            "per_clip": [{"clip_id": dataset.clip_id(i), "frames": n_frames[i], "labeled": n_lab[i],
                          "group": dataset.clip_group(i)} for i in range(len(dataset))],
        }
        print(f"  clips: {len(dataset)}   frames: {sum(n_frames)}   labelled frames: {sum(n_lab)}")
        if n_frames:
            print(f"  frames per clip   {{count: clips}}: {_fmt_counter(Counter(n_frames))}")
            print(f"  labelled per clip {{count: clips}}: {_fmt_counter(Counter(n_lab))}")
            print(f"  labelled positions inside clips {{pos: clips}}: {_fmt_counter(positions, 20)}")
            print(f"  groups (e.g. source sequences): {len({dataset.clip_group(i) for i in range(len(dataset))})}")
            for i in range(min(5, len(dataset))):
                print(f"    clip {dataset.clip_id(i)}: {n_frames[i]} frames, {n_lab[i]} labelled "
                      f"at {np.flatnonzero(dataset.labeled_mask(i)).tolist()}")
        if not len(dataset):
            rep.problem(f"[{split}] no clip found")
        elif sum(n_lab) and min(n_lab) == 0:
            rep.problem(f"[{split}] {n_lab.count(0)} clips have no labelled frame")

    with rep.section("resolution"):
        frames = [(ci, t) for ci in range(len(dataset)) for t in range(dataset.num_frames(ci))]
        if args.max_resolution_frames:
            frames = _evenly(frames, args.max_resolution_frames)
        sizes = Counter(dataset.frame_size(ci, t) for ci, t in frames)
        result["frame_sizes"] = {f"{h}x{w}": n for (h, w), n in sizes.items()}
        print(f"  frame sizes over {len(frames)} frames {{HxW: frames}}: "
              f"{_fmt_counter(Counter({f'{h}x{w}': n for (h, w), n in sizes.items()}))}")
        if len(sizes) > 1:
            rep.problem(f"[{split}] frames have {len(sizes)} different sizes")

    labelled = _labelled_frames(dataset)
    sample = _evenly(labelled, args.sample if args.sample is not None else int(cfg.get("sample_annotations", 20)))
    hist = np.zeros(dataset.num_classes + 1, dtype=np.int64)  # last bin = ignore
    labels_cache: Dict[Tuple[int, int], np.ndarray] = {}
    with rep.section(f"annotation ids on {len(sample)} sampled annotations"):
        result["samples"] = []
        for ci, t in sample:
            entry: Dict[str, Any] = {"clip_id": dataset.clip_id(ci), "position": t,
                                     "frame_index": int(dataset.frame_indices(ci)[t])}
            try:
                label = dataset.load_label(ci, t)
                labels_cache[(ci, t)] = label
                valid = (label != dataset.ignore_index) & (label < dataset.num_classes)
                hist[:-1] += np.bincount(label[valid].astype(np.int64), minlength=dataset.num_classes)[: dataset.num_classes]
                hist[-1] += int((~valid).sum())
                entry["label_shape"] = list(label.shape)
                if tuple(label.shape) != dataset.frame_size(ci, t):
                    rep.problem(f"[{split}] {entry['clip_id']}@{t}: label {label.shape} != frame {dataset.frame_size(ci, t)}")
                ids = dataset.annotation_ids(ci, t)
                if ids is None:
                    print(f"  {entry['clip_id']} pos {t}: adapter exposes no raw ids; classes "
                          f"{sorted(np.unique(label).tolist())}")
                else:
                    uniq, counts = np.unique(ids, return_counts=True)
                    rows = dataset.explain_annotation_ids(ci, t, uniq)
                    area = dict(zip(uniq.tolist(), counts.tolist()))
                    for r in rows:
                        r["pixels"] = int(area.get(r["id"], 0))
                    entry["ids"] = rows
                    unmapped = [r["id"] for r in rows if r.get("category") is None]
                    print(f"  {entry['clip_id']} pos {t} (frame {entry['frame_index']}): {len(uniq)} unique ids")
                    for r in rows[:40]:
                        print(f"      id {r['id']:>9} -> category {str(r.get('category')):>4} -> train id "
                              f"{r['train_id']:>3} ({r['class']}), {r['pixels']} px")
                    if len(rows) > 40:
                        print(f"      ... {len(rows) - 40} more ids")
                    if unmapped:
                        print(f"      ids without a category (void): {unmapped[:10]}")
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                rep.problem(f"[{split}] {entry['clip_id']} pos {t}: {entry['error']}")
            result["samples"].append(entry)

    with rep.section("class histogram over the sampled annotations (train classes + ignore)"):
        total = int(hist.sum())
        names = list(dataset.class_names) + ["<ignore>"]
        result["histogram"] = {n: int(c) for n, c in zip(names, hist)}
        if total:
            for name, count in zip(names, hist):
                print(f"  {name:15s} {100 * count / total:6.2f}%  {int(count):>12} px")
            if hist[-1] / total > 0.9:
                rep.problem(f"[{split}] {100 * hist[-1] / total:.0f}% of sampled pixels are ignore: check the label mapping")
        else:
            print("  (no labelled frame sampled)")

    for alt in cfg.get("crosscheck", []) or []:
        if not labels_cache:
            break
        with rep.section(f"cross-check vs {alt.get('name')}"):
            alt_spec = dict(spec, overrides={**(spec.get("overrides") or {}), **alt.get("overrides", {})})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                alt_ds = build_dataset(alt_spec, base_dir=REPO_ROOT)
            for w in caught:
                print(f"  warning: {w.message}")
            main_dir = dataset.diagnostics().get("index", {}).get("annotation_dir")
            alt_dir = alt_ds.diagnostics().get("index", {}).get("annotation_dir")
            if main_dir is not None and main_dir == alt_dir:
                print("  same source as the configured one: skipped")
                continue
            lookup = {(alt_ds.clip_id(i), int(f)): (i, t) for i in range(len(alt_ds))
                      for t, f in enumerate(alt_ds.frame_indices(i))}
            agree = total_px = compared = 0
            pairs: Counter = Counter()
            for (ci, t), label in labels_cache.items():
                key = (dataset.clip_id(ci), int(dataset.frame_indices(ci)[t]))
                if key not in lookup:
                    continue
                other = alt_ds.load_label(*lookup[key])
                if other is None or other.shape != label.shape:
                    continue
                compared += 1
                agree += int((other == label).sum())
                total_px += label.size
                diff = other != label
                pairs.update(zip(label[diff].tolist(), other[diff].tolist()))
            names = list(dataset.class_names)
            fmt = lambda c: names[c] if 0 <= c < len(names) else "ignore"  # noqa: E731
            top = [(f"{fmt(a)}->{fmt(b)}", n) for (a, b), n in pairs.most_common(8)]
            result.setdefault("crosscheck", {})[alt.get("name")] = {
                "frames_compared": compared, "pixel_agreement": agree / total_px if total_px else None,
                "top_disagreements": top,
            }
            if not compared:
                print("  no common labelled frame to compare")
            else:
                print(f"  {compared} frames: {100 * agree / total_px:.2f}% of pixels agree")
                for text, n in top:
                    print(f"      {text}: {n} px ({100 * n / total_px:.2f}%)")

    if overlays_left > 0 and labels_cache:
        with rep.section(f"overlays ({out_dir})"):
            by_clip: Dict[int, Tuple[int, int]] = {}
            for ci, t in labels_cache:
                by_clip.setdefault(ci, (ci, t))
            picks = _evenly(list(by_clip.values()), overlays_left)
            out_dir.mkdir(parents=True, exist_ok=True)
            result["overlays"] = []
            for ci, t in picks:
                clip_id = dataset.clip_id(ci)
                frame_no = int(dataset.frame_indices(ci)[t])
                title = f"{dataset.name} {split} clip {clip_id} frame {frame_no} (pos {t})"
                img = compose_overlay(dataset.load_frame(ci, t), labels_cache[(ci, t)], dataset.palette,
                                      dataset.ignore_index, dataset.class_names, title)
                path = out_dir / f"overlay_{split}_{clip_id}_{frame_no:06d}.png"
                img.save(path)
                result["overlays"].append(str(path))
                print(f"  saved {path}")
                overlays_left -= 1
    return result, overlays_left


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rep = Report()
    cfg = expand_env(apply_overrides(load_yaml(args.config), args.set))
    spec = dict(cfg.get("dataset") or {})
    if not spec.get("type"):
        print(f"[diagnose] {args.config} needs a dataset.type", file=sys.stderr)
        return 2
    if args.fixture:
        spec["root"] = str(ensure_fixture())
    elif args.root:
        spec["root"] = args.root
    splits = args.splits or cfg.get("splits") or ["val"]
    overlays_left = args.overlays if args.overlays is not None else int(cfg.get("overlays", 3))
    out_dir = Path(args.out)
    print(f"[diagnose] config {args.config}; dataset spec {spec}; splits {splits}")

    report: Dict[str, Any] = {"config": cfg, "dataset_spec": spec, "splits": {}}
    for split in splits:
        try:
            result, overlays_left = diagnose_split(split, spec, cfg, args, rep, out_dir, overlays_left)
        except Exception as exc:  # last-resort guard: keep going with the next split
            rep.problem(f"[{split}] unexpected failure: {type(exc).__name__}: {exc}")
            result = {"split": split, "error": str(exc)}
        report["splits"][split] = result

    report["problems"] = rep.problems
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "diagnose_report.json"
    from eval.results import to_jsonable

    path.write_text(json.dumps(to_jsonable(report), indent=1))
    print("\n================ summary ================")
    if rep.problems:
        print(f"{len(rep.problems)} problem(s):")
        for p in rep.problems:
            print(f"  - {p}")
    else:
        print("no problem detected")
    print(f"full report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
