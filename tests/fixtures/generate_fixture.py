"""Generate the tiny synthetic Cityscapes-VPS fixture used by the unit tests.

This writes a handful of small PNGs that mimic the directory layout and naming
documented at the top of ``src/data/cityscapes_vps.py`` — a *toy* stand-in so
the loader can be tested with no real (license-gated) dataset present.

The generated PNGs are tiny and fully deterministic, so they are NOT committed
to the repo (see .gitignore); instead the test suite regenerates them on demand
via tests/conftest.py -> ``ensure_fixture()``. Run this directly only if you
want the files on disk to inspect them, or to (re)generate them by hand:

    python tests/fixtures/generate_fixture.py

Toy layout produced (root = tests/fixtures/cityscapes_vps):

    val/
        img_all/                 clip 0000: frames 0..5   (6 frames)
                                 clip 0001: frames 0..2   (3 frames)
        panoptic_video/          panoptic PNGs for labeled frames only
                                 (frame_index % 5 == 0)  -> 0000:{0,5}, 0001:{0}
        labelmap/                single-channel semantic trainId PNGs, same frames

This first ("legacy") fixture follows the earlier, unverified layout guess
(clip-prefixed names inside img_all) and backs tests/test_cityscapes_vps.py.
A second fixture, ``cityscapes_vps_official/`` (see ``generate_official``),
mirrors the layout verified against the official VPSNet preparation scripts
and backs the tests of configs/cityscapes_vps.yaml.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

FIXTURE_ROOT = Path(__file__).resolve().parent / "cityscapes_vps"
SPLIT = "val"
CITY = "frankfurt"
SEQ = 0  # original Cityscapes sequence id
HEIGHT, WIDTH = 8, 16
ANNOTATED_EVERY = 5

# clip id -> number of frames in that clip
CLIPS = {0: 6, 1: 3}


def _frame_name(clip: int, frame: int, suffix: str, ext: str = ".png") -> str:
    # e.g. 0000_0003_frankfurt_000000_000003_leftImg8bit.png
    return f"{clip:04d}_{frame:04d}_{CITY}_{SEQ:06d}_{frame:06d}{suffix}{ext}"


def _make_frame_rgb(clip: int, frame: int) -> np.ndarray:
    """A small, deterministic, visually-distinct RGB frame."""
    rng = np.random.default_rng(clip * 100 + frame)
    img = rng.integers(0, 60, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    # A bright moving square so successive frames look different in the overlay.
    x = (frame * 2) % (WIDTH - 3)
    img[2:5, x : x + 3] = np.array([220, 40 + 30 * clip, 40], dtype=np.uint8)
    return img


def _make_panoptic_rgb() -> np.ndarray:
    """COCO-panoptic-encoded RGB: id = R + 256*G + 256*256*B. Two segments."""
    ann = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    ann[:, : WIDTH // 2, 0] = 1  # left half  -> id 1
    ann[:, WIDTH // 2 :, 0] = 2  # right half -> id 2
    return ann


def _make_semantic_labelmap() -> np.ndarray:
    """Single-channel semantic trainId map (toy: road=0 / building=2)."""
    lbl = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    lbl[HEIGHT // 2 :, :] = 2
    return lbl


def generate(root: Path = FIXTURE_ROOT) -> Path:
    """(Re)write the fixture under ``root`` and return the split directory."""
    img_dir = root / SPLIT / "img_all"
    pan_dir = root / SPLIT / "panoptic_video"
    lbl_dir = root / SPLIT / "labelmap"
    for d in (img_dir, pan_dir, lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_img = n_ann = 0
    for clip, n_frames in CLIPS.items():
        for frame in range(n_frames):
            Image.fromarray(_make_frame_rgb(clip, frame), mode="RGB").save(
                img_dir / _frame_name(clip, frame, "_leftImg8bit")
            )
            n_img += 1
            if frame % ANNOTATED_EVERY == 0:  # labeled frame
                Image.fromarray(_make_panoptic_rgb(), mode="RGB").save(
                    pan_dir / _frame_name(clip, frame, "_gtFine_panoptic")
                )
                Image.fromarray(_make_semantic_labelmap(), mode="L").save(
                    lbl_dir / _frame_name(clip, frame, "_gtFine_labelTrainIds")
                )
                n_ann += 1

    return root / SPLIT


def ensure_fixture(root: Path = FIXTURE_ROOT) -> Path:
    """Generate the fixture only if it is missing; return the dataset root.

    Idempotent and safe to call from conftest / scripts on every run.
    """
    if not (root / SPLIT / "img_all").is_dir():
        generate(root)
    return root


# =========================================================================== #
# Official-layout fixture
# =========================================================================== #
# A second toy dataset that mirrors the layout and encodings VERIFIED against
# the official VPSNet preparation scripts (see the header of
# src/data/cityscapes_vps.py), used to test configs/cityscapes_vps.yaml:
#
#   <root>/panoptic_gt_val_city_vps.json        COCO-panoptic GT for val/panoptic_video
#   <root>/panoptic_im_<split>_city_vps.json    image lists (img/ names)
#   <root>/im_all_info_<val|test>_city_vps.json image lists (img_all/ names)
#   <root>/<split>/img/        labelled frames, {vid}_{fid}_{city}_{seq}_{frame}_{newImg8bit|leftImg8bit}.png
#   <root>/<val|test>/img_all/ all frames, original names {city}_{seq}_{frame}_leftImg8bit.png
#   <root>/<train|val>/labelmap/       uint8 train ids, 255 void (crowd keeps its class)
#   <root>/<train|val>/panoptic_inst/  uint16: stuff = train id, thing = id*1000+obj, 255 void+crowd
#   <root>/val/panoptic_video/         RGB ids (IdGenerator-like colours), class only in the JSON
#
# Clips have 10 frames (real: 30) labelled at positions 4 and 9 (real: 4..29
# step 5); position 9 plays the original Cityscapes "20th frame" (gtFine names).
OFFICIAL_ROOT = Path(__file__).resolve().parent / "cityscapes_vps_official"
OFFICIAL_VERSION = "2"  # bump to force regeneration when the layout changes
OH, OW = 16, 32
GT_POSITION = 9

# (train id, name, colour, isthing) as in prepare_data/city_default.py.
OFFICIAL_CATEGORIES = [
    (0, "road", (128, 64, 128), 0), (1, "sidewalk", (244, 35, 232), 0),
    (2, "building", (70, 70, 70), 0), (3, "wall", (102, 102, 156), 0),
    (4, "fence", (190, 153, 153), 0), (5, "pole", (153, 153, 153), 0),
    (6, "traffic light", (250, 170, 30), 0), (7, "traffic sign", (220, 220, 0), 0),
    (8, "vegetation", (107, 142, 35), 0), (9, "terrain", (152, 251, 152), 0),
    (10, "sky", (70, 130, 180), 0), (11, "person", (220, 20, 60), 1),
    (12, "rider", (255, 0, 0), 1), (13, "car", (0, 0, 142), 1),
    (14, "truck", (0, 0, 70), 1), (15, "bus", (0, 60, 100), 1),
    (16, "train", (0, 80, 100), 1), (17, "motorcycle", (0, 0, 230), 1),
    (18, "bicycle", (119, 11, 32), 1),
]
_COLOUR = {c[0]: c[2] for c in OFFICIAL_CATEGORIES}
_ISTHING = {c[0]: c[3] for c in OFFICIAL_CATEGORIES}

# (split, vps video id, city, seq, first Cityscapes frame, n frames, labelled positions)
OFFICIAL_CLIPS = [
    ("val", "0001", "frankfurt", 0, 275, 10, (4, 9)),
    ("val", "0002", "frankfurt", 0, 557, 10, (4, 9)),  # same sequence as 0001
    ("val", "0003", "lindau", 1, 10, 10, (4, 9)),
    ("train", "0004", "munster", 2, 10, 10, (4, 9)),
    ("train", "0005", "munster", 3, 10, 10, (4, 9)),
    ("test", "0006", "lindau", 4, 10, 10, (4, 9)),
]


def official_names(vid: str, city: str, seq: int, frame: int, pos: int) -> Dict[str, str]:
    """File names of one frame in img_all/, img/ and the annotation folders."""
    cs = f"{city}_{seq:06d}_{frame:06d}"
    gt = pos == GT_POSITION
    return {
        "img_all": f"{cs}_leftImg8bit.png",
        "img": f"{vid}_{pos:04d}_{cs}_{'leftImg8bit' if gt else 'newImg8bit'}.png",
        "ann": f"{vid}_{pos:04d}_{cs}_{'gtFine_color' if gt else 'final_mask'}.png",
        "key": cs,
    }


def official_scene(clip_no: int, pos: int) -> Tuple[np.ndarray, np.ndarray]:
    """(semantic train ids with crowd kept, object id per thing pixel or -1)."""
    sem = np.full((OH, OW), 255, dtype=np.uint8)
    inst = np.full((OH, OW), -1, dtype=np.int64)
    sem[0:4, :] = 10  # sky
    sem[4:9, :] = 2  # building
    sem[4:9, 20:22] = 5  # pole
    sem[9:16, :] = 0  # road
    sem[9:16, 0:6] = 1  # sidewalk
    x0 = 6 + pos  # a car (obj 1) moving right over time
    sem[9:13, x0 : x0 + 6] = 13
    inst[9:13, x0 : x0 + 6] = 1
    sem[5:12, 24:27] = 11  # a person whose object id is 0 (allowed by the VPS script)
    inst[5:12, 24:27] = 0
    if clip_no % 2 == 1:  # a second car instance in some clips
        sem[13:16, 20:28] = 13
        inst[13:16, 20:28] = 2
    sem[13:15, 8:12] = 18  # crowd of bicycles: class, but no object id
    sem[15, 28:32] = 255  # void (e.g. ego vehicle)
    return sem, inst


def official_expected(clip_no: int, pos: int, source: str) -> np.ndarray:
    """Expected train-id map: 'labelmap' keeps crowd, 'panoptic' marks it void."""
    sem, inst = official_scene(clip_no, pos)
    if source == "labelmap":
        return sem.copy()
    out = sem.copy()
    crowd = (np.vectorize(lambda c: _ISTHING.get(int(c), 0))(sem) == 1) & (inst < 0)
    out[crowd] = 255
    return out


class _IdColours:
    """Deterministic stand-in for panopticapi.utils.IdGenerator."""

    def __init__(self):
        self.taken = {(0, 0, 0)} | {_COLOUR[c] for c in _COLOUR if not _ISTHING[c]}

    def colour(self, cat: int) -> Tuple[int, int, int]:
        base = _COLOUR[cat]
        if not _ISTHING[cat]:
            return base
        k = 0
        while True:
            col = tuple(int(np.clip(b + d * k, 0, 255)) for b, d in zip(base, (7, 5, -6)))
            if col not in self.taken:
                self.taken.add(col)
                return col  # type: ignore[return-value]
            k += 1


def _rgb_id(col: Tuple[int, int, int]) -> int:
    return int(col[0] + 256 * col[1] + 65536 * col[2])


def _official_frame(sem: np.ndarray, clip_no: int, pos: int) -> np.ndarray:
    palette = np.full((256, 3), 20, dtype=np.int64)  # void -> dark grey
    for cat, col in _COLOUR.items():
        palette[cat] = col
    rng = np.random.default_rng(1000 + clip_no * 50 + pos)
    img = palette[sem] + rng.integers(-8, 9, size=(OH, OW, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=1))


def generate_official(root: Path = OFFICIAL_ROOT) -> Path:
    """(Re)write the official-layout fixture under ``root``; return ``root``."""
    if root.exists():
        shutil.rmtree(root)
    categories = [
        {"id": c, "name": n, "isthing": t, "color": list(col)} for c, n, col, t in OFFICIAL_CATEGORIES
    ]
    images: Dict[str, List[Dict]] = {}
    images_all: Dict[str, List[Dict]] = {}
    gt_annotations: Dict[str, List[Dict]] = {}
    id_colours: Dict[str, _IdColours] = {}
    running_id = 0
    for clip_no, (split, vid, city, seq, first, n_frames, labelled) in enumerate(OFFICIAL_CLIPS):
        split_dir = root / split
        colours = id_colours.setdefault(split, _IdColours())
        inst2colour: Dict[Tuple[int, int], Tuple[int, int, int]] = {}  # reset per video
        for pos in range(n_frames):
            frame = first + pos
            names = official_names(vid, city, seq, frame, pos)
            sem, inst = official_scene(clip_no, pos)
            rgb = _official_frame(sem, clip_no, pos)
            if split in ("val", "test"):
                (split_dir / "img_all").mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb, mode="RGB").save(split_dir / "img_all" / names["img_all"])
                running_id += 1
                images_all.setdefault(split, []).append(
                    {"id": running_id, "file_name": names["img_all"], "width": OW, "height": OH}
                )
            if pos not in labelled:
                continue
            (split_dir / "img").mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb, mode="RGB").save(split_dir / "img" / names["img"])
            image_id = names["img"].rsplit("_", 1)[0]
            images.setdefault(split, []).append(
                {"id": image_id, "file_name": names["img"], "width": OW, "height": OH}
            )
            if split == "test":
                continue  # test labels are withheld
            # labelmap: train ids, crowd keeps its class.
            (split_dir / "labelmap").mkdir(parents=True, exist_ok=True)
            Image.fromarray(sem, mode="L").save(split_dir / "labelmap" / names["ann"])
            # panoptic_inst: stuff id / thing id*1000+obj / 255 for void and crowd.
            pan_inst = np.full((OH, OW), 255, dtype=np.uint16)
            for cat in np.unique(sem):
                cat = int(cat)
                if cat == 255:
                    continue
                m = sem == cat
                if not _ISTHING[cat]:
                    pan_inst[m] = cat
                else:
                    pan_inst[m & (inst >= 0)] = cat * 1000 + inst[m & (inst >= 0)]
            (split_dir / "panoptic_inst").mkdir(parents=True, exist_ok=True)
            Image.fromarray(pan_inst).save(split_dir / "panoptic_inst" / names["ann"])
            if split != "val":
                continue  # the official script builds panoptic_video for val only
            # panoptic_video: COCO RGB ids; class per segment only in the JSON.
            pan_rgb = np.zeros((OH, OW, 3), dtype=np.uint8)
            segments = []
            for value in np.unique(pan_inst):
                value = int(value)
                if value == 255:
                    continue
                cat, obj = (value // 1000, value % 1000) if value >= 1000 else (value, -1)
                if (cat, obj) not in inst2colour:
                    inst2colour[(cat, obj)] = colours.colour(cat)
                col = inst2colour[(cat, obj)]
                m = pan_inst == value
                pan_rgb[m] = col
                segments.append({"id": _rgb_id(col), "category_id": cat, "iscrowd": 0, "area": int(m.sum())})
            (split_dir / "panoptic_video").mkdir(parents=True, exist_ok=True)
            Image.fromarray(pan_rgb, mode="RGB").save(split_dir / "panoptic_video" / names["ann"])
            gt_annotations.setdefault(split, []).append(
                {"image_id": image_id, "file_name": names["ann"], "segments_info": segments}
            )

    for split, ims in images.items():
        _write_json(root / f"panoptic_im_{split}_city_vps.json", {"images": ims, "categories": categories})
    for split, ims in images_all.items():
        _write_json(root / f"im_all_info_{split}_city_vps.json", {"images": ims, "categories": categories})
    for split, anns in gt_annotations.items():
        _write_json(
            root / f"panoptic_gt_{split}_city_vps.json",
            {"images": images[split], "annotations": anns, "categories": categories},
        )
    (root / ".fixture_version").write_text(OFFICIAL_VERSION)
    return root


def ensure_official_fixture(root: Path = OFFICIAL_ROOT) -> Path:
    """Generate the official-layout fixture if missing or stale; return its root."""
    marker = root / ".fixture_version"
    if not marker.is_file() or marker.read_text().strip() != OFFICIAL_VERSION:
        generate_official(root)
    return root


def main() -> None:
    generate()
    generate_official()
    for root in (FIXTURE_ROOT, OFFICIAL_ROOT):
        n_files = sum(1 for _ in root.rglob("*.png"))
        print(f"Wrote {n_files} PNG files under {root}")


if __name__ == "__main__":
    main()
