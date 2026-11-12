# Method 5, phase 1: one-pass warp bound — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, on VPS train, whether a frozen model on an oracle-warped frame beats the same model on the uniformly downscaled frame of the same size, and read the pre-registered pass rule.

**Architecture:** A small pure-torch module builds a separable sampling grid from a saliency map, warps a frame and unwarps logits. A second module makes the three oracle saliencies from a teacher prediction. A per-clip runner does the network passes on the labelled frames and returns one confusion matrix per configuration. A CLI (`bound`, `merge`, `score`) mirrors `scripts/method4.py` and reuses its file handling and its reading.

**Tech Stack:** Python, PyTorch (`grid_sample`), numpy, pytest. No new dependency.

**Spec:** `docs/superpowers/specs/2026-11-12-method5-warp-bound-design.md` (accepted 2026-11-12; frozen after the first `bound` run).

## Global Constraints

- Frozen backbones, fp32 accuracy, `train_all` only. `bound` refuses `--split val` unless `--fixture`.
- Grid, fixed: c ∈ {0.25, 0.375, 0.5, 0.75}; saliency ∈ {E, B, S}; α ∈ {0.25, 0.5, 0.75}; σ ∈ {1, 4} cells; timing ∈ {now, prev}. 144 configurations per model, plus 4 uniform (α = 0) rows.
- Saliency grid: cells of 16 px (64 × 128 on a 1024 × 2048 frame). Density clip: [0.5, 1/c].
- Small classes (train ids): pole 5, traffic light 6, traffic sign 7, person 11, rider 12, motorcycle 17, bicycle 18.
- Latency of a configuration: `area` frame at scale c + `reuse` `post`. Resampling, unwarping and saliency cost 0.
- Pass rule: best configuration with timing `now` ≥ +0.010 mIoU above the primary full-frame curve at equal mean latency, on all 400 clips, for at least one backbone. `PASS_MARGIN = 0.010`.
- Pipeline check: each α = 0 row within 0.001 mIoU of the full-frame baseline at the same scale; otherwise no verdict.
- Small commits; `pytest -q` green before every push. Method 2, 3 and 4 results are not touched.
- The user runs or approves every GPU job. Every try goes in `results/method5/TRIES.md`.
- Python: `C:/tmp/avrh-venv/Scripts/python.exe`. Branch: `warp-bound`.

## Review Focus

1. **A saliency that is all zeros** (no error, no small class in the frame): the grid must be exactly the uniform one, not NaN. Test in Task 1.
2. **A saliency concentrated in one cell at c = 0.25 and α = 0.75:** the clip to [0.5, 4] and the renormalisation must end with mean 1 and every value inside the clip. Test in Task 1.
3. **A model whose logits use `align_corners=True` or an output stride above 1:** the α = 0 unwarp must still equal the baseline's `upsample_logits`. Test in Task 1, both conventions.
4. **A bound file started with another grid, or cut off mid-clip:** `bound` must refuse the first and drop the tail of the second, as `method4 bound` does. Test in Task 4.
5. **A partial clip set at `score`:** the verdict must be "not read", never pass or fail. Test in Task 4.

---

### Task 1: The separable warp (`src/warp/grid.py`)

**Files:**
- Create: `src/warp/__init__.py`, `src/warp/grid.py`
- Test: `tests/test_warp_grid.py`

**Interfaces:**
- Produces:
  - `axis_density(profile: Tensor[G], alpha: float, scale: float, d_min: float = 0.5) -> Tensor[G]`
  - `forward_coords(density: Tensor[G], n_out: int) -> Tensor[n_out]`: source coordinate in [0, 1] sampled by each output pixel centre.
  - `inverse_coords(density: Tensor[G], n_src: int) -> Tensor[n_src]`: output coordinate in [0, 1] of each source pixel centre.
  - `@dataclass WarpGrid: dy: Tensor, dx: Tensor` and `make_grid(saliency: Tensor[gh, gw], alpha: float, scale: float, d_min: float = 0.5) -> WarpGrid`
  - `warp_frame(frame: Tensor[1,3,H,W], grid: WarpGrid, out_size: Tuple[int,int], supersample: int = 2) -> Tensor[1,3,H',W']`
  - `unwarp_logits(logits: Tensor[1,C,h,w], grid: WarpGrid, frame_size: Tuple[int,int], align_corners: bool) -> Tensor[1,C,H,W]`

The maths the signatures do not determine:

```
profile  p  = mean of the saliency over the other axis
density  d  = (1 - alpha) + alpha * p / mean(p)        (all ones if mean(p) == 0)
           repeat up to 50 times: d = clip(d, d_min, 1/scale); d = d / mean(d)
           stop when the clip changes nothing (max change < 1e-6)
cdf      F  : piecewise linear over the G cells, F(0) = 0, F(1) = 1, slope d
forward     : x = F^-1(u) at u = (j + 0.5) / n_out
inverse     : u = F(x)    at x = (i + 0.5) / n_src
```

`warp_frame` samples at `supersample × out_size` with `grid_sample` (bilinear, `align_corners=False`, `padding_mode="border"`), then `avg_pool2d(supersample)`. `unwarp_logits` samples the logits once, at the inverse coordinates, with the model's corner convention.

- [ ] **Step 1: Write the failing tests** in `tests/test_warp_grid.py`:

```python
def test_alpha_zero_or_empty_saliency_is_uniform():
    for sal, a in ((torch.rand(4, 8), 0.0), (torch.zeros(4, 8), 0.75)):
        g = make_grid(sal, a, 0.5)
        assert torch.allclose(g.dx, torch.ones(8)) and torch.allclose(g.dy, torch.ones(4))
    assert torch.allclose(forward_coords(torch.ones(8), 16), (torch.arange(16) + 0.5) / 16)

def test_density_respects_the_clip_and_has_mean_one():
    p = torch.zeros(128); p[40] = 1.0                      # one hot cell
    for scale in (0.25, 0.375, 0.5, 0.75):
        d = axis_density(p, 0.75, scale)
        assert d.min() >= 0.5 - 1e-5 and d.max() <= 1 / scale + 1e-5
        assert abs(float(d.mean()) - 1) < 1e-5 and torch.isfinite(d).all()
        assert d.argmax() == 40

def test_coords_are_monotone_and_inverse_undoes_forward():
    d = axis_density(torch.rand(32), 0.75, 0.5)
    f, inv = forward_coords(d, 64), inverse_coords(d, 256)
    assert (f[1:] > f[:-1]).all() and (inv[1:] > inv[:-1]).all()
    # source x -> output u -> back to x through the forward map's interpolation
    x = (torch.arange(256) + 0.5) / 256
    back = torch.from_numpy(np.interp(inv.numpy(), (np.arange(64) + 0.5) / 64, f.numpy()))
    assert (back - x)[8:-8].abs().max() < 2 / 64

def test_a_salient_block_is_magnified():
    sal = torch.zeros(4, 8); sal[1, 2] = 1.0
    g = make_grid(sal, 0.75, 0.5)
    frame = torch.zeros(1, 3, 64, 128); frame[..., 16:32, 32:48] = 1.0   # the salient cell
    warped = warp_frame(frame, g, (32, 64))
    assert warped.mean() > 1.5 * frame.mean()              # the block takes a larger share of the output

def test_alpha_zero_warp_is_the_antialiased_downscale():
    frame = torch.rand(1, 3, 64, 128)
    frame = F.avg_pool2d(frame, 5, 1, 2)                   # smooth content
    ref = foveation.ops.downsample(frame, 0.5)
    out = warp_frame(frame, make_grid(torch.zeros(4, 8), 0.0, 0.5), (32, 64))
    assert (out - ref).abs().max() < 0.02

@pytest.mark.parametrize("align", [False, True])
def test_alpha_zero_unwarp_is_the_baseline_upsampling(align):
    logits = torch.randn(1, 19, 8, 16)
    ref = F.interpolate(logits, size=(64, 128), mode="bilinear", align_corners=align)
    out = unwarp_logits(logits, make_grid(torch.zeros(4, 8), 0.0, 0.5), (64, 128), align)
    assert torch.allclose(out, ref, atol=1e-5)
```

- [ ] **Step 2: Run** `pytest tests/test_warp_grid.py -q`. Expected: import error for `warp.grid`.
- [ ] **Step 3: Implement** the six functions and the dataclass in `src/warp/grid.py`.
- [ ] **Step 4: Run** `pytest tests/test_warp_grid.py -q`. Expected: 7 passed.
- [ ] **Step 5: Commit** `Method 5 phase 1: separable warp grid, warp and unwarp`.

### Task 2: The oracle saliencies (`src/warp/oracle.py`)

**Files:**
- Create: `src/warp/oracle.py`
- Test: `tests/test_warp_oracle.py`

**Interfaces:**
- Produces:
  - `SMALL_CLASSES = (5, 6, 7, 11, 12, 17, 18)`, `SAL_CELL = 16`
  - `saliency(kind: str, teacher: Tensor[H,W] long, uniform: Optional[Tensor[H,W]] , sigma: float, cell: int = SAL_CELL) -> Tensor[H/cell, W/cell] float`; `kind` ∈ {"E", "B", "S"}; "E" needs `uniform` (raises `ValueError` without it); an unknown kind raises.

Pixel masks: E `uniform != teacher`; B a pixel whose label differs from its right or lower neighbour, marked on both pixels; S `teacher` in `SMALL_CLASSES`. Then `avg_pool2d(cell)` and a Gaussian of `sigma` cells (radius `ceil(3·sigma)`, reflect padding, kernel normalised).

- [ ] **Step 1: Write the failing tests:**

```python
def test_each_kind_marks_the_right_pixels():
    t = torch.zeros(32, 64, dtype=torch.long); t[:, 32:] = 1; t[0:16, 0:16] = 5   # a pole block
    u = t.clone(); u[16:32, 48:64] = 3                                             # the uniform pass is wrong there
    e, b, s = (saliency(k, t, u, sigma=0.0 + 1e-6, cell=16) for k in "EBS")
    assert e.shape == (2, 4) and e.argmax() == 7 and e.flatten()[:7].sum() < 1e-3
    assert s.argmax() == 0 and s.flatten()[1:].sum() < 1e-3
    assert b[0, 0] > 0 and b[1, 1] > 0 and b[1, 3] < 1e-3                          # block edge, class edge, flat area

def test_smoothing_spreads_and_keeps_the_mass():
    t = torch.zeros(64, 128, dtype=torch.long); t[16:32, 48:64] = 11
    a, b = saliency("S", t, None, 1e-6, 16), saliency("S", t, None, 1.0, 16)
    assert (b > 0).sum() > (a > 0).sum() and abs(float(a.sum() - b.sum())) < 0.05 * float(a.sum())

def test_bad_inputs_raise():
    t = torch.zeros(16, 32, dtype=torch.long)
    with pytest.raises(ValueError): saliency("E", t, None, 1.0)
    with pytest.raises(ValueError): saliency("X", t, t, 1.0)
```

- [ ] **Step 2: Run** `pytest tests/test_warp_oracle.py -q`. Expected: import error.
- [ ] **Step 3: Implement** `saliency` in `src/warp/oracle.py`.
- [ ] **Step 4: Run** the tests. Expected: 3 passed.
- [ ] **Step 5: Commit** `Method 5 phase 1: oracle saliencies (error, boundaries, small classes)`.

### Task 3: The per-clip runner and the latency (`src/warp/bound.py`)

**Files:**
- Create: `src/warp/bound.py`
- Modify: `src/foveation/latency_model.py` (add `predict_warp_frame` after `predict_canvas_frame`)
- Test: `tests/test_warp_bound.py`, `tests/test_latency_model.py` (append)

**Interfaces:**
- Consumes: Task 1 and 2; `reuse.method.full_pass(model, frame, scale, small, dtype)` (labels as the full-frame baseline); `eval.harness.frame_to_tensor`, `FullFrameMethod`; `reuse.replay._confusion_t`; `models.utils.upsample_logits`.
- Produces:
  - `@dataclass(frozen=True) WarpConfig(scale: float, saliency: Optional[str] = None, alpha: float = 0.0, sigma: float = 1.0, timing: str = "now")`; `name()` gives `5o/c0.5/E/a0.5/s4/now`, and `5o/c0.5/uniform` when `alpha == 0`. `__post_init__` raises `ValueError` for alpha outside [0, 1), an unknown saliency or timing, a saliency given with alpha 0, or none given with alpha > 0.
  - `warped_pass(model, frame: Tensor[1,3,H,W], grid: WarpGrid, scale: float, dtype) -> Tensor[H,W] long`: warp to `round(scale·H) × round(scale·W)`, one forward, `unwarp_logits` with `model.align_corners`, argmax.
  - `bound_clip(dataset, index: int, model, cfgs: Sequence[WarpConfig], device, allow_any_frame_size: bool = False, cell: int = SAL_CELL) -> Dict[str, np.ndarray]`: configuration name → (19, 19) int64 confusion summed over the clip's labelled frames. Per labelled frame t: teacher and uniform passes of t and of t − 1 (t itself if t == 0) with `full_pass`; one `warped_pass` per configuration. Refuses a frame that is not 1024 × 2048 unless `allow_any_frame_size`.
  - `LatencyTables.predict_warp_frame(model: str, scale: float) -> Dict[str, Any]` with keys `median_ms`, `p95_ms`, `gflops`, `parts`: the `area` frame entry at `scale` plus the `reuse` `post` entry, read as `predict_canvas_frame` reads its context part. An unmeasured scale raises the usual missing-run error.

- [ ] **Step 1: Write the failing tests** (`vps`, `oracle_backbone` fixtures from `tests/conftest.py`; saliency cell 8 on the 16 × 32 fixture):

```python
def test_names_and_bad_configs():
    assert WarpConfig(0.5, "E", 0.5, 4, "now").name() == "5o/c0.5/E/a0.5/s4/now"
    assert WarpConfig(0.375, "S", 0.25, 1, "prev").name() == "5o/c0.375/S/a0.25/s1/prev"
    assert WarpConfig(0.75).name() == "5o/c0.75/uniform"
    for bad in (dict(alpha=0.5), dict(saliency="E"), dict(saliency="Q", alpha=0.5),
                dict(saliency="E", alpha=1.0), dict(saliency="E", alpha=0.5, timing="next")):
        with pytest.raises(ValueError): WarpConfig(0.5, **bad)

def test_uniform_rows_equal_the_full_frame_baseline(vps, oracle_backbone):
    cfgs = [WarpConfig(0.5), WarpConfig(1.0)]
    out = bound_clip(vps, 0, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=8)
    for cfg in cfgs:   # the same confusion as full_pass at that scale on the labelled frames
        assert np.array_equal(out[cfg.name()], _baseline_confusion(vps, 0, oracle_backbone, cfg.scale))

def test_every_config_is_scored_on_every_labelled_pixel(vps, oracle_backbone):
    cfgs = [WarpConfig(0.5, k, 0.75, 1, tm) for k in "EBS" for tm in ("now", "prev")] + [WarpConfig(0.5)]
    out = bound_clip(vps, 0, oracle_backbone, cfgs, "cpu", allow_any_frame_size=True, cell=8)
    totals = {int(v.sum()) for v in out.values()}
    assert set(out) == {c.name() for c in cfgs} and len(totals) == 1 and totals.pop() > 0

def test_a_real_size_is_required(vps, oracle_backbone):
    with pytest.raises(ValueError, match="1024"):
        bound_clip(vps, 0, oracle_backbone, [WarpConfig(0.5)], "cpu")
```

`_baseline_confusion` is a test helper: `full_pass` labels against the labels, with `_confusion_t`. If the scale-0.5 equality cannot hold exactly on the fixture (the 2 × supersampled warp is not bit-identical to the antialiased resize), keep the exact assertion for scale 1.0 and assert at scale 0.5 that the two confusions differ on fewer than 2 % of the pixels; say so in the test's docstring.

Append to `tests/test_latency_model.py`: `test_warp_frame_sums_area_and_post` (the two parts add up to `median_ms`, for a table built as in `test_canvas_frame_sums_its_parts`) and `test_unmeasured_warp_scale_names_the_missing_run` (scale 0.6 raises `KeyError`).

- [ ] **Step 2: Run** `pytest tests/test_warp_bound.py tests/test_latency_model.py -q`. Expected: import errors.
- [ ] **Step 3: Implement** `WarpConfig`, `warped_pass`, `bound_clip`, `predict_warp_frame`.
- [ ] **Step 4: Run** the same command. Expected: all pass.
- [ ] **Step 5: Commit** `Method 5 phase 1: per-clip runner and T4 latency of a warped pass`.

### Task 4: The CLI (`scripts/method5.py`)

**Files:**
- Create: `scripts/method5.py`
- Test: `tests/test_method5_cli.py`

**Interfaces:**
- Consumes: `bound_clip`, `WarpConfig`, `predict_warp_frame`; from `scripts.method4`: `resume`, `_append`, `e2e_curves`, `DEFAULT_E2E`; from `scripts.method3`: `_dataset_and_clips`, `_tag`, `read_lines`, `fullframe_ref`, `judge_readings`, `_public`, `DEFAULT_LATENCY`; from `scripts.method2`: `_floats`, `_strs`, `latency_tables`, `model_config`.
- Produces three subcommands, with the common flags of `method4` (`--split`, `--models`, `--clip-ids`, `--chunk`, `--tag`, `--out-dir`, `--fixture`, `--device`):
  - `bound`: flags `--scales 0.25,0.375,0.5,0.75 --saliencies E,B,S --alphas 0.25,0.5,0.75 --sigmas 1,4 --timings now,prev --cell 16`. Writes `experiments/method5/<split>/bound_<split>_<model>_<tag>.jsonl.gz`, one gzip member per clip, one line per configuration: `{split, model, clip_id, name, config, conf}`. Resumable per clip; refuses another grid; refuses `val` without `--fixture`. Prints seconds per clip.
  - `merge`: as `method4 merge` (refuses duplicates and unequal configuration counts, lists missing clips).
  - `score`: writes `score_<split>_<model>.json` and `verdict_<split>.json`. Per configuration: `miou`, `per_class_iou`, `latency`, `judges` (mean, p95), `judges_e2e`, `delta_vs_uniform` (against the α = 0 row of the same scale), `budget_ms` (latency saved at equal mIoU, mean judge). Top level: `fullframe`, `fullframe_e2e`, `pipeline_check` (per scale: the α = 0 mIoU, the baseline mIoU, `ok` when they differ by at most 0.001), `best` per saliency and timing, `verdict`.
  - Verdict: `passes` is `None` when the clip set is incomplete or a pipeline check is not ok (the message says which); otherwise true when the best `now` configuration's ΔmIoU at equal mean latency is ≥ `PASS_MARGIN`.

- [ ] **Step 1: Write the failing tests** in `tests/test_method5_cli.py`, on the fixture (`--fixture --cell 8 --scales 0.5 --saliencies E,S --alphas 0.5 --sigmas 1`), following `tests/test_method4_cli.py`:
  - `test_bound_writes_one_line_per_config_and_clip`: 2 clips → (4 + 1) × 2 lines; every `conf` is 19 × 19.
  - `test_bound_resumes_and_skips_finished_clips`: a second run adds no line.
  - `test_bound_drops_a_cut_off_tail`: append half a gzip member, run again, the file reads cleanly with the same lines.
  - `test_bound_refuses_another_grid`: same tag, `--alphas 0.25` → `SystemExit` naming the tag.
  - `test_bound_refuses_val_without_fixture`.
  - `test_merge_refuses_a_clip_merged_twice`.
  - `test_score_on_a_partial_set_does_not_read_the_rule`: one clip of two → `verdict["passes"] is None`, message contains "partial".
  - `test_score_reads_the_rule_on_the_complete_set`: `passes` is a bool and `pipeline_check` has one entry per scale.
  - `test_verdict_needs_the_margin`: call the verdict function on hand-made configurations with Δ = 0.009 and 0.011 → False, True; a `prev` configuration at +0.05 alone → False.
  - `test_a_failed_pipeline_check_blocks_the_verdict`: hand-made check with a 0.002 gap → `passes is None`.
- [ ] **Step 2: Run** `pytest tests/test_method5_cli.py -q`. Expected: import error.
- [ ] **Step 3: Implement** `scripts/method5.py`.
- [ ] **Step 4: Run** `pytest -q`. Expected: everything passes (310 + the new tests), 5 skipped.
- [ ] **Step 5: Commit** `Method 5 phase 1: CLI bound (resumable per clip), merge, score with the pipeline check and the pass rule`.

### Task 5: Results pages and the Colab data path

**Files:**
- Create: `results/method5/README.md`, `results/method5/TRIES.md`, `notebooks/method5_bound_colab.ipynb`
- Modify: `scripts/make_train_img_all.py` (add `--zip-labelled-prev`)
- Test: `tests/test_make_train_img_all.py` (create or append, whichever exists)

**Interfaces:**
- Produces: `zip_labelled_prev(root: Path, out: Path) -> Path`, writing `vps_train_labelled_prev.zip`: for each train clip, the labelled frames and the frame before each, from `train/img_all`, with `train/panoptic_inst` and the root train JSONs, at the same paths as the chunk zips.

- [ ] **Step 1: Check how the `train_all` split lists frames.** Read `src/data/` for the split. If it needs all 30 frames of a clip to exist on disk, the zip cannot be used as is: record that in TRIES and make the notebook use the four chunk zips instead (skip steps 2 and 3).
- [ ] **Step 2: Write the failing test** `test_zip_labelled_prev_holds_two_frames_per_label`: on a temporary root with 2 clips of 10 frames and labels at 4 and 9, the zip holds frames 3, 4, 8, 9 of each clip, the label files and the JSONs, and nothing else.
- [ ] **Step 3: Implement** `zip_labelled_prev` and the flag; run the test.
- [ ] **Step 4: Write** `results/method5/README.md` (question, mechanism, grid, reading, pass rule, PowerShell run commands, "Results: pending") and `results/method5/TRIES.md` (first entry: spec accepted 2026-11-12, the three plan details).
- [ ] **Step 5: Write** the notebook, as `notebooks/method4_bound_colab.ipynb`: first cell `CHUNK`, `MODELS`, `BRANCH = "warp-bound"`; data cell reads the zip chosen in step 1; run cell calls `scripts.method5 bound` and copies the bound file to `MyDrive/avrh/method5/train_all/` every 10 clips.
- [ ] **Step 6: Run** `pytest -q`, then **commit** `Method 5 phase 1: results pages, Colab notebook, labelled+previous frames zip`.

### Task 6: Runs (each one approved by the user)

- [ ] **Step 1: Pipeline check on 5 clips**, PIDNet-S, laptop: `python -m scripts.method5 bound --split train_all --clip-ids <5 ids> --models pidnet_s --alphas 0.5 --saliencies E --sigmas 1 --timings now --tag pipecheck`, then `score --no-expect` on that file. Expected: the four α = 0 rows within 0.001 mIoU of the baseline on those clips. If not: stop, log, ask the user (spec). Log seconds per clip and correct the runtime estimate in the README.
- [ ] **Step 2: Chunk 0** (100 clips, both models, full grid), laptop or Colab. `merge`, `score`: partial, the rule is not read. Log in TRIES.
- [ ] **Step 3: Chunks 1 to 3**, then `merge` and `score` on the 400 clips.

### Gate

- [ ] Read `verdict_train_all.json` against the pre-registered rule.
- [ ] Copy the verdict and the gzipped scores into `results/method5/`, write the result in `results/method5/README.md` and TRIES, update the top-level `README.md`.
- [ ] **Fails:** stop and ask the user. **Passes:** phase 2 spec and plan, reviewed by the user before any code.
