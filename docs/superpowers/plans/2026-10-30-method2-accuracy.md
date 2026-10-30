# Method 2 Accuracy Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the mIoU of memory-free foveation (low-resolution context + k full-resolution glimpses) for PIDNet-S, BiSeNetV2 and the mixed pair, and place every configuration against the full-frame curve at equal T4 latency.

**Architecture:** Cells of the glimpse grid don't overlap, and a cell's glimpse doesn't depend on the other cells. So per frame we cache, for every cell, the confusion matrix of the context prediction and of the glimpse prediction, plus the context features the selection rules need. Any selection is then evaluated exactly by summing per-cell matrices. Latency comes from the three T4 latency-floor sessions, through a parts model. A real-pipeline test checks the cache against `FoveatedSegmenter` run through the harness.

**Tech Stack:** Python, PyTorch, NumPy, the existing `eval` harness, `foveation` ops and `scripts/latency_floor.py`.

**Spec:** `docs/superpowers/specs/2026-10-30-method2-accuracy-design.md`

## Global Constraints

- Accuracy metric: standard 19-class mIoU only. Per-class IoUs are reported; no class subset is ever an objective.
- Accuracy pass in fp32 (as the baselines). One check that fp16 changes mIoU by < 0.1.
- Latency comes only from the T4 latency-floor tables (CUDA graph level, median of the three session medians) or a T4 run. Never from the laptop.
- Tune on VPS train; val is evaluated once, on finalists chosen from train.
- Search grid = measured T4 points: context scale ∈ {0.25, 0.375, 0.5}, glimpse size ∈ {256, 384, 512}, margin 32, k ∈ {1, 2, 4, 8}, selection ∈ {topk, greedy (suppress 1)}. The overhead table has no k = 16, so k = 16 needs a new latency-floor run first.
- Frame size 1024×2048 (the latency tables' frame).
- Everything deterministic: fixed seeds for random selection and the bootstrap.
- The foveation forward stays free of host syncs (no `.item()`, no device-scalar indexing), so it remains CUDA-graph capturable.
- Pairs: (pidnet_s, pidnet_s), (bisenetv2, bisenetv2), (bisenetv2 context, pidnet_s glimpses).

## Review Focus

1. **A frame that isn't 1024×2048** (another dataset, or a resized copy): cache building and latency prediction must refuse it with a message naming the size, rather than silently using tables measured at another size. Pinned in Task 4.
2. **Cells without valid pixels** (all ignore, or entirely in the bottom/right canvas padding): their per-cell matrices must be zero and their oracle gain 0, never a crash or a negative count. Pinned in Task 4.
3. **A class absent from a bootstrap resample**: mIoU must skip it the way `ConfusionMatrix.compute` does (NaN excluded), not count it as 0. Pinned in Task 7.
4. **An off-grid configuration** (k = 3, margin 16, scale 0.3): latency prediction must raise an error saying a `latency_floor` run is needed for that point. Pinned in Task 6.
5. **An interrupted cache build**: a half-written cache must never be loaded as complete; a rerun rebuilds it. Pinned in Task 8.

---

### Task 1: Selection-score variants in the foveation ops

**Files:**
- Modify: `src/foveation/ops.py` (replace `entropy_map` use and `cell_scores`)
- Modify: `src/foveation/pipeline.py` (`FoveationConfig`, `FoveatedSegmenter.forward`)
- Test: `tests/test_foveation.py`

**Interfaces:**
- Produces:
  - `score_map(logits: Tensor (B,C,h,w), kind: str) -> Tensor (B,h,w)` fp32; kind ∈ `SCORE_KINDS = ("entropy", "margin", "maxprob")`. Margin = `1 - (p1 - p2)`, maxprob = `1 - p1`.
  - `samples_per_cell(map_h: int, grid: GlimpseGrid) -> int`, written `q` below. It is `max(1, round(map_h * grid.inner / grid.frame_size[0]))`.
  - `cell_samples(m: Tensor (h,w), grid, q, mode: str) -> Tensor (cells, q*q)`. It resizes `m` to `(rows*q, cols*q)` with the same row/column scaling to the padded extent as today's `cell_scores`. `mode="bilinear"` (align_corners False) is for scores, `"nearest"` for class ids. Bins are now equal-sized by construction.
  - `cell_features(ctx_logits: Tensor (1,C,h,w), grid, kinds: Sequence[str]) -> Dict[str, Tuple[Tensor (cells,C), Tensor (cells,)]]`. Per kind it returns `class_sums` (the sum of the score over a cell's samples, split by the context argmax class at each sample, taken with nearest sampling) and `top10` (the mean of the top `ceil(0.1*q*q)` samples of the cell).
  - `pool_scores(class_sums, top10, q: int, pool: str, class_weights: Optional[Tensor (C,)]) -> Tensor (cells,)`:
    - `pool="mean"`: `class_sums.sum(1) / q²`, or `(class_sums @ w) / q²` with weights;
    - `pool="top10"`: `top10`. Passing weights with top10 raises `ValueError`.
  - `select_greedy_batch(scores: Tensor (N,cells), grid, k, suppress) -> Tensor (N,k)`. Row i must equal `select_greedy(scores[i], …)`.
  - `FoveationConfig` gains `score: str = "entropy"`, `pool: str = "mean"`, `class_weights: Optional[Tuple[float, ...]] = None`.
  - `FoveatedSegmenter.forward(frame, return_idx: bool = False)` returns `labels`, or `(labels, idx)` when `return_idx` is True.
  - `cell_scores(score_map, grid)` stays as a thin wrapper (mean pooling through `cell_samples`), for existing callers.

- [ ] **Step 1: Write the failing tests**

```python
def test_score_kinds_on_known_logits():
    logits = torch.tensor([[[[10.0]], [[0.0]], [[0.0]]]])  # confident pixel, C=3
    assert score_map(logits, "maxprob").item() < 1e-3
    assert score_map(logits, "margin").item() < 1e-3
    flat = torch.zeros(1, 3, 1, 1)
    assert torch.isclose(score_map(flat, "entropy"), torch.tensor(math.log(3))).all()
    assert torch.isclose(score_map(flat, "margin"), torch.tensor(1.0)).all()

def test_pool_mean_with_unit_weights_equals_plain_mean():
    grid = GlimpseGrid((1024, 2048), 384, 32)
    logits = torch.randn(1, 19, 128, 256, generator=torch.Generator().manual_seed(0))
    feats = cell_features(logits, grid, ["entropy"])
    sums, top = feats["entropy"]
    q = samples_per_cell(128, grid)
    plain = pool_scores(sums, top, q, "mean", None)
    ones = pool_scores(sums, top, q, "mean", torch.ones(19))
    assert torch.allclose(plain, ones, atol=1e-6)
    assert torch.allclose(plain, cell_scores(score_map(logits, "entropy")[0], grid), atol=1e-5)
    assert (top >= plain - 1e-6).all()  # top-10% mean >= cell mean

def test_top10_with_class_weights_is_rejected():
    with pytest.raises(ValueError):
        pool_scores(torch.zeros(28, 19), torch.zeros(28), 10, "top10", torch.ones(19))

def test_greedy_batch_matches_row_by_row():
    grid = GlimpseGrid((1024, 2048), 256, 32)
    s = torch.rand(5, grid.num_cells, generator=torch.Generator().manual_seed(1))
    batch = select_greedy_batch(s, grid, 4, 1)
    for i in range(5):
        assert torch.equal(batch[i], select_greedy(s[i], grid, 4, 1))

@pytest.mark.parametrize("score,pool,weights", [("margin", "mean", None), ("maxprob", "top10", None),
                                                ("entropy", "mean", tuple([2.0] + [1.0] * 18))])
def test_segmenter_variants_run_without_host_sync(score, pool, weights):
    # same pattern as the existing sync-free test: set_sync_debug_mode("error") on CUDA, else CPU run
    ...
```

Add the variants to the existing sync-free test rather than writing a second harness.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_foveation.py -q`
Expected: FAIL with ImportError on `score_map`.

- [ ] **Step 3: Implement the ops and the config fields**

In `forward`, compute `cell_features(ctx_logits, grid, [cfg.score])`, then `pool_scores(...)`, then select. Keep `class_weights` as a non-persistent buffer built in `__init__`, so `forward` does no host-to-device copy. Use `torch.topk` (fixed k) for the top-10% mean. `select_greedy_batch` is the vectorised form of `select_greedy`: the same `-1e9` suppression and the same `finfo.min` mask, on (N, cells).

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass. The existing `cell_scores` tests may need new expected values, because the bins are now equal-sized; update them and say so in the commit message.

- [ ] **Step 5: Commit**

```bash
git add src/foveation/ops.py src/foveation/pipeline.py tests/test_foveation.py
git commit -m "Foveation: entropy/margin/maxprob scores, mean/top-10% pooling, class weights"
```

### Task 2: `FoveatedMethod`, the harness wrapper

**Files:**
- Create: `src/foveation/method.py`
- Modify: `src/foveation/__init__.py` (export)
- Test: `tests/test_foveation_method.py`

**Interfaces:**
- Consumes: `FoveationConfig`, `FoveatedSegmenter` (Task 1); `eval.harness.FrameMethod`, `evaluate`; `eval.cost.count_flops`.
- Produces: `FoveatedMethod(backbone_ctx, backbone_glimpse, cfg: FoveationConfig, precision: str = "fp32")` with:
  - `name = "foveated"`, `requires_all_frames = False`;
  - `predict(frame, t, ledger) -> (H,W) long`, which builds the segmenter lazily for the frame size;
  - `predict_with_idx(frame) -> (labels, idx)`, used by the Task 5 check;
  - `latency_fns(frame_size, device) -> {"end_to_end": fn}`;
  - `describe()`, which returns the `asdict(cfg)` fields plus `pixel_fraction`, `grid`, `context_model` and `glimpse_model` (the backbones' class names), plus `precision`.
  - Ledger components per frame:
    - `context` (FLOPs of the context backbone at its input; pixels `h*w` of the context input);
    - `glimpses` (FLOPs of the glimpse backbone on (k,3,size,size); pixels `k*size²`);
    - `policy` and `recomposition` (flops 0, pixels 0).

  FLOPs are counted once per frame size, as in `FullFrameMethod`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture(scope="module")
def vps():  # the official-layout fixture, as in tests/test_harness.py
    return CityscapesVPS(CityscapesVPSConfig.from_yaml(CONFIG, root=str(gf.OFFICIAL_ROOT), split="val"))

def _oracle_backbone(vps):  # nearest-colour dummy at stride 1: perfect at full resolution
    return DummyBackbone(num_classes=19, output_stride=1, class_colors=vps.palette / 255.0).eval()

def test_every_cell_glimpsed_is_exact_and_costs_are_split(vps):
    bb = _oracle_backbone(vps)
    grid = GlimpseGrid((gf.OH, gf.OW), 64, 8)
    cfg = FoveationConfig(context_scale=0.25, glimpse_size=64, glimpse_margin=8, k=grid.num_cells)
    m = FoveatedMethod(bb, bb, cfg)
    assert isinstance(m, FrameMethod)
    res = evaluate(vps, m)
    assert res["metrics"]["miou"] == pytest.approx(1.0)  # tiling + paste lose nothing
    comps = res["cost"]["components"]
    assert set(comps) == {"context", "glimpses", "policy", "recomposition"}
    assert comps["glimpses"]["pixels_per_frame"] == grid.num_cells * 64 * 64

def test_foveated_method_matches_the_segmenter(vps):
    bb = _oracle_backbone(vps)
    cfg = FoveationConfig(context_scale=0.25, glimpse_size=64, glimpse_margin=8, k=2)
    frame = frame_to_tensor(vps.load_frame(0, int(np.flatnonzero(vps.labeled_mask(0))[0])), "cpu")
    m = FoveatedMethod(bb, bb, cfg)
    labels, idx = m.predict_with_idx(frame)
    ref_labels, ref_idx = FoveatedSegmenter(bb, bb, cfg, (gf.OH, gf.OW))(frame, return_idx=True)
    assert torch.equal(labels, ref_labels) and torch.equal(idx, ref_idx)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_foveation_method.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `FoveatedMethod`**

For fp16, deep-copy the backbones as `FullFrameMethod` does. Compute in the backbones' dtype and do argmax and pasting on fp32 logits, as in the segmenter.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_foveation_method.py tests/test_harness.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foveation/method.py src/foveation/__init__.py tests/test_foveation_method.py
git commit -m "FoveatedMethod: foveation as a FrameMethod with per-component costs"
```

### Task 3: Per-clip full-frame reference curves (train and val)

**Files:**
- Create: `scripts/method2_fullframe.py`
- Test: `tests/test_method2_fullframe.py`

**Interfaces:**
- Consumes: `scripts.common` (config, overrides, device), `data.registry.build_dataset`, `models.registry.build_backbone`, `FullFrameMethod`, `ConfusionMatrix`.
- Produces: `python -m scripts.method2_fullframe --config configs/baseline_fullframe_<model>.yaml --split {train,val} [--max-clips N] [--fixture] --out experiments/method2/fullframe`. It writes `fullframe_<backbone.type>_<split>.npz` with:
  - `scales (S,) float64`, in the config's `eval.scales` order;
  - `confusion (S, N_clips, C, C) int64`;
  - `clip_ids (N_clips,) str`;
  - `meta` (JSON string: config, git commit, dataset fingerprint).

  It loads each frame once and runs every scale on it.

- [ ] **Step 1: Write the failing test.** On the fixture with the dummy backbone (`configs/baseline_fixture.yaml`, `--fixture`), the npz exists. Summing `confusion[s]` over clips gives the same `miou` as `scripts.run_baseline` for that scale, to 1e-9.
- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_method2_fullframe.py -q`
Expected: FAIL (no module).

- [ ] **Step 3: Implement the script.** Follow `scripts/run_baseline.run` for config and dataset loading. Set the split with `cfg["dataset"]["split"] = args.split`. Refuse any frame that isn't 1024×2048 unless `--fixture`.
- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/test_method2_fullframe.py -q`
Expected: PASS.

- [ ] **Step 5: Real-data check on val.** Use the venv at `C:\tmp\avrh-venv`, with `CITYSCAPES_VPS_ROOT` and `PIDNET_S_WEIGHTS` set. Run both models on val, then on train in the background (about 2,400 frames × 5 scales per model). Expected: the val mIoU per scale equals the README table (PIDNet-S 71.5 / 70.1 / 64.8 / 58.6 / 50.1; BiSeNetV2 70.4 / 69.9 / 67.6 / 64.3 / 59.1) to ±0.05. The train run must load 400 clips × 6 labelled frames; if the adapter fails on train, fix the `split_overrides.train` block in `configs/cityscapes_vps.yaml`.
- [ ] **Step 6: Check the BiSeNetV2 checkpoint's training split.** In CoinCheung/BiSeNet's README, `model_final_v2_city.pth` must have been trained on Cityscapes train only. Write the finding, with its source line, into the Task 11 results README. If it was trained on train+val, stop and tell the user.
- [ ] **Step 7: Commit**

```bash
git add scripts/method2_fullframe.py tests/test_method2_fullframe.py
git commit -m "Method 2: per-clip full-frame reference confusions on train and val"
```

### Task 4: Per-cell caches

**Files:**
- Create: `src/foveation/cache.py`
- Test: `tests/test_foveation_cache.py`

**Interfaces:**
- Consumes: `GlimpseGrid`, `downsample`, `gather_crops`, `cell_features`, `samples_per_cell`, `SCORE_KINDS` (Task 1); `models.utils.upsample_logits`.
- Produces:
  - `per_cell_confusion(pred: Tensor (H,W), label: Tensor (H,W), grid, num_classes, ignore_index) -> Tensor (cells,C,C) int32`. It covers only the inner square of each cell, and only its in-frame pixels. Pixels whose label is `ignore_index` are excluded. It uses one `bincount` over `cell*C*C + t*C + p`.
  - `context_cells(backbone, frame (1,3,H,W), label, scale, grid, num_classes, ignore_index) -> ContextCells`: a dataclass with `conf (cells,C,C) int32`, `class_sums {kind: (cells,C) float32}`, `top10 {kind: (cells,) float32}` and `q: int`, for all `SCORE_KINDS`. The context labels are made exactly as in `FoveatedSegmenter.forward`.
  - `glimpse_cells(backbone, frame, label, grid, num_classes, ignore_index, chunk: int = 16) -> Tensor (cells,C,C) int32`. Every cell is glimpsed, in chunks, and only the inner square is scored, exactly as `paste_labels` would place it.
  - `build_cache(dataset, kind: str, backbone, model_name: str, grid, out_path: Path, device, scale: Optional[float] = None, clip_indices=None) -> Path`. `kind` ∈ {"context", "glimpse"}. It writes an npz with:
    - `conf (N,cells,C,C) int32`, `clip_index (N,) int32`, `t (N,) int32`;
    - for context, also `class_sums_<kind> (N,cells,C) float32`, `top10_<kind> (N,cells) float32` and `q`;
    - `meta` (JSON: kind, model, scale, size, margin, split, git commit, dataset fingerprint, frame size, rows, cols).

    It writes to `<out>.tmp.npz` and renames it on completion. It refuses frames that aren't 1024×2048 unless `allow_any_frame_size=True`, which the fixture tests use.
  - `cache_path(root: Path, split, kind, model, grid, scale=None) -> Path`:
    - context: `root/<split>/ctx_<model>_s<scale>_g<size>m<margin>.npz`;
    - glimpse: `root/<split>/gl_<model>_g<size>m<margin>.npz`.

- [ ] **Step 1: Write the failing tests**

```python
def test_per_cell_confusion_sums_to_the_frame_confusion():
    g = torch.Generator().manual_seed(0)
    grid = GlimpseGrid((100, 230), 48, 8)  # rows/cols with padding at the bottom/right
    pred = torch.randint(0, 5, (100, 230), generator=g)
    label = torch.randint(0, 5, (100, 230), generator=g)
    label[:10] = 255
    cells = per_cell_confusion(pred, label, grid, 5, 255)
    cm = ConfusionMatrix(5, 255); cm.update(pred.numpy(), label.numpy())
    assert np.array_equal(cells.sum(0).numpy(), cm.matrix)
    assert (cells >= 0).all()

def test_cells_in_padding_or_all_ignore_are_zero():
    grid = GlimpseGrid((100, 230), 48, 8)
    label = torch.full((100, 230), 255); label[:32, :32] = 1
    cells = per_cell_confusion(torch.ones(100, 230, dtype=torch.long), label, grid, 5, 255)
    assert cells[0].sum() == 32 * 32 and cells[1:].sum() == 0

def test_cache_refuses_other_frame_sizes(tmp_path, vps):  # vps / _oracle_backbone as in Task 2's tests
    with pytest.raises(ValueError, match="1024x2048"):
        build_cache(vps, "glimpse", _oracle_backbone(vps), "dummy", GlimpseGrid((1024, 2048), 384, 32),
                    tmp_path / "x.npz", "cpu")

def test_glimpse_cache_of_a_perfect_backbone_is_diagonal(tmp_path, vps):
    grid = GlimpseGrid((gf.OH, gf.OW), 64, 8)
    p = build_cache(vps, "glimpse", _oracle_backbone(vps), "dummy", grid, tmp_path / "gl.npz", "cpu",
                    allow_any_frame_size=True)
    conf = np.load(p)["conf"].sum((0, 1))
    assert conf.sum() > 0 and conf.sum() == np.trace(conf)
```

Move `vps` and `_oracle_backbone` into `tests/conftest.py`, so Tasks 2, 4, 5 and 8–10 share them.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_foveation_cache.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `cache.py`**
- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_foveation_cache.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foveation/cache.py tests/test_foveation_cache.py
git commit -m "Method 2: per-cell context and glimpse confusion caches"
```

### Task 5: Cached evaluation of selection rules, checked against the real pipeline

**Files:**
- Create: `src/foveation/cached_eval.py`
- Test: `tests/test_cached_eval.py`

**Interfaces:**
- Consumes: the Task 4 caches; `pool_scores`, `select_greedy_batch`, `GlimpseGrid` (Task 1); `ConfusionMatrix`.
- Produces:
  - `CachedPair.load(ctx_path, gl_path) -> CachedPair`, with `ctx_conf`, `gl_conf`, `features`, `q`, `grid`, `clip_index`, `t` and `meta`. It asserts the same frames, the same order and the same grid.
  - `Rule`, a frozen dataclass:
    - `kind: str`, one of "uncertainty", "oracle", "random", "prior" or "all";
    - `score="entropy"`, `pool="mean"`, `class_weights: Optional[Tuple[float, ...]] = None`;
    - `selection="topk"`, `suppress=1`, `seed=0`;
    - `name() -> str`, a stable id for JSON.
  - `select_cells(pair, rule, k, prior_order: Optional[Tensor (cells,)] = None) -> Tensor (N,cells) bool`:
    - uncertainty: `pool_scores` per frame, then a stable top-k sort, or `select_greedy_batch` for greedy;
    - oracle: the top-k cells by `diag(gl).sum - diag(ctx).sum`, using a stable sort;
    - random: `torch.randperm` with `torch.Generator().manual_seed(rule.seed + frame index)`;
    - prior: the same `prior_order[:k]` cells in every frame;
    - all: every cell (k is ignored).
  - `prior_order(pair) -> Tensor (cells,)`: cells sorted by mean oracle gain over the pair's frames. Always computed on train and applied to any split.
  - `compose(pair, mask) -> np.ndarray (n_clips, C, C) int64`, the per-clip confusion: context matrices for unselected cells, glimpse matrices for selected ones.
  - `miou_of(matrix: np.ndarray (C,C)) -> float`, identical to `ConfusionMatrix.compute()["miou"]` (no unpredicted pixels exist here).

- [ ] **Step 1: Write the failing unit tests**
  - With fixture caches: `compose` with an all-False mask equals the context-only confusion, and with `Rule("all")` it equals the confusion of the fully-tiled prediction.
  - `oracle` mIoU ≥ `uncertainty` mIoU ≥ … isn't guaranteed, so test only that the oracle's total correct pixels are ≥ those of any rule at the same k.
  - A random rule is reproducible for a given seed.
- [ ] **Step 2: Write the real-pipeline check.** Skip it unless CUDA is available and `CITYSCAPES_VPS_ROOT`, `PIDNET_S_WEIGHTS` and the BiSeNetV2 weights are present. It covers the first 2 val clips and three configurations:
  - PIDNet-S, context 0.25, 384, k 4, topk entropy mean;
  - the same with greedy;
  - BiSeNetV2 context + PIDNet-S glimpses, context 0.25, 384, k 4, topk.

  Build the caches for those clips into `tmp_path`. Run `FoveatedMethod.predict_with_idx` frame by frame on the same frames. Assert that:
  - the selected cell ids are identical on every frame;
  - per clip, `|compose − pipeline confusion|.sum() ≤ 1e-4 × valid pixels`. The glimpse batch size differs (chunk vs k), so cuDNN may flip a few argmax pixels;
  - the mIoU differs by < 0.02 points.
- [ ] **Step 3: Run them to verify they fail**

Run: `python -m pytest tests/test_cached_eval.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 4: Implement `cached_eval.py`**
- [ ] **Step 5: Run the tests, including the real-data check on the laptop**

Run: `python -m pytest tests/test_cached_eval.py -q`
Expected: PASS, with the real-data test executed rather than skipped.

- [ ] **Step 6: Commit**

```bash
git add src/foveation/cached_eval.py tests/test_cached_eval.py
git commit -m "Method 2: exact cached evaluation of selection rules, checked against the pipeline"
```

### Task 6: Latency model from the T4 tables

**Files:**
- Create: `src/foveation/latency_model.py`
- Test: `tests/test_latency_model.py`

**Interfaces:**
- Consumes: `results/latency_floor/latency_floor_t4-s{1,2,3}.json`.
- Produces:
  - `LatencyTables.load(paths) -> LatencyTables` (graph level only). Per record name, it keeps the median over sessions of `median_ms` and of `p95_ms`, plus `gflops`.
  - `LatencyTables.from_records(records: Mapping[str, list])`, for tests.
  - `.fullframe(model, scale) -> {"median_ms", "p95_ms", "gflops"}`.
  - `.predict(ctx_model, ctx_scale, glimpse_model, size, k, selection) -> {"median_ms", "p95_ms", "gflops", "parts": {...}}`:
    - median = context area median + glimpse batch median + overhead median;
    - p95 = the sum of the parts' p95 (conservative; say so in the README);
    - gflops = context + glimpses;
    - overhead stride = 1 if either model is BiSeNetV2, else 8.
  - Off-grid requests raise `KeyError` with the message "no T4 measurement for <part> <args>: run scripts/latency_floor.py for this point".
  - `STRIDE = {"pidnet_s": 8, "bisenetv2": 1}`.

- [ ] **Step 1: Write the failing tests**

```python
TABLES = LatencyTables.load(sorted(Path("results/latency_floor").glob("latency_floor_t4-s*.json")))

def test_prediction_matches_every_measured_e2e_point():
    for r in graph_e2e_records():  # graph-level e2e records, median over sessions
        p = TABLES.predict(r["context_model"], r["context_scale"], r["glimpse_model"], r["size"], r["k"], "topk")
        assert abs(r["median_ms"] / p["median_ms"] - 1) < 0.05, r["name"]

def test_fullframe_values():
    assert TABLES.fullframe("pidnet_s", 1.0)["median_ms"] == pytest.approx(19.35, abs=0.01)
    assert TABLES.fullframe("bisenetv2", 0.5)["median_ms"] == pytest.approx(8.06, abs=0.01)

def test_off_grid_point_names_the_missing_run():
    with pytest.raises(KeyError, match="latency_floor"):
        TABLES.predict("pidnet_s", 0.25, "pidnet_s", 384, 3, "topk")
```

  If a mixed-pair point misses the 5% band, don't widen it: report the gap to the user and add a mixed overhead row to `latency_floor`.
- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_latency_model.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `latency_model.py`**
- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_latency_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/foveation/latency_model.py tests/test_latency_model.py
git commit -m "Method 2: T4 latency model from the latency-floor sessions"
```

### Task 7: Trade-off readings and clip bootstrap

**Files:**
- Create: `src/eval/tradeoff.py`
- Test: `tests/test_tradeoff.py`

**Interfaces:**
- Consumes: `miou_of` (Task 5).
- Produces:
  - `FullFrameCurve(latency_ms: Sequence[float], miou: Sequence[float])`:
    - `miou_at(latency) -> float`: piecewise linear in latency; NaN outside the range;
    - `latency_at(miou) -> float`: the inverse; NaN outside the range, or if mIoU isn't strictly increasing with latency.
  - `delta_miou_at_equal_latency(miou, latency_ms, curve) -> float` = `miou - curve.miou_at(latency_ms)`.
  - `latency_saved_at_equal_miou(miou, latency_ms, curve) -> float` = `curve.latency_at(miou) - latency_ms`.
  - `clip_bootstrap(config_clips: (N,C,C), ff_clips: (S,N,C,C), ff_latency_ms: (S,), latency_ms: float, n: int = 1000, seed: int = 0) -> Dict[str, Tuple[float, float]]`. It returns 95% percentile intervals of `miou`, `delta_miou` and `latency_saved`. One clip resample is shared by the configuration and every reference scale. NaN resamples are dropped, and their count is reported as `nan_resamples`.

- [ ] **Step 1: Write the failing tests**

```python
def test_curve_interpolation():
    c = FullFrameCurve([2.0, 10.0], [0.50, 0.70])
    assert c.miou_at(6.0) == pytest.approx(0.60)
    assert c.latency_at(0.65) == pytest.approx(8.0)
    assert math.isnan(c.miou_at(12.0)) and math.isnan(c.latency_at(0.75))

def test_absent_class_is_skipped_like_confusion_matrix():
    m = np.zeros((3, 3), np.int64); m[0, 0] = 5; m[1, 1] = 3; m[1, 0] = 1  # class 2 absent
    cm = ConfusionMatrix(3); cm.matrix = m
    assert miou_of(m) == pytest.approx(cm.compute()["miou"])

def test_bootstrap_of_the_reference_itself_brackets_zero():
    rng = np.random.default_rng(0)
    clips = rng.integers(0, 50, (20, 4, 4))
    ff = np.stack([clips // 2 + np.eye(4, dtype=int) * 10, clips])  # two scales
    # the configuration *is* the reference scale at 10 ms: every resample gives delta 0
    ci = clip_bootstrap(clips, ff, np.array([2.0, 10.0]), 10.0, n=200, seed=0)
    assert ci["delta_miou"] == pytest.approx((0.0, 0.0), abs=1e-12)
    assert ci == clip_bootstrap(clips, ff, np.array([2.0, 10.0]), 10.0, n=200, seed=0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_tradeoff.py -q`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `tradeoff.py`**
- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_tradeoff.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/eval/tradeoff.py tests/test_tradeoff.py
git commit -m "Trade-off readings at equal latency / equal mIoU with clip bootstrap"
```

### Task 8: `scripts/method2.py cache`

**Files:**
- Create: `scripts/method2.py` (subcommand `cache`)
- Test: `tests/test_method2_cli.py`

**Interfaces:**
- Consumes: `build_cache`, `cache_path` (Task 4); `scripts.common`; the `backbone` sections of `configs/baseline_fullframe_{pidnet_s,bisenetv2}.yaml`.
- Produces: `python -m scripts.method2 cache --split {train,val} [--models pidnet_s,bisenetv2] [--scales 0.25,0.375,0.5] [--sizes 256,384,512] [--margin 32] [--root experiments/method2/cache] [--max-clips N] [--fixture]`. It builds one context cache per (model, scale, size) and one glimpse cache per (model, size). It skips files that already exist, but never `*.tmp.npz` files, which it deletes and rebuilds. Fixture mode uses the dummy backbone with `allow_any_frame_size=True`.

- [ ] **Step 1: Write the failing tests**
  - In fixture mode, two scales and one size write the expected file names.
  - A second run builds nothing.
  - A planted `<name>.tmp.npz` is rebuilt, and the final file exists.
- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_method2_cli.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement the `cache` subcommand** (argparse subparsers, so Tasks 9 and 10 add theirs).
- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_method2_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/method2.py tests/test_method2_cli.py
git commit -m "Method 2 CLI: build per-cell caches"
```

### Task 9: `sweep` and `tune-weights`

**Files:**
- Modify: `scripts/method2.py`
- Test: `tests/test_method2_cli.py`

**Interfaces:**
- Consumes: `CachedPair`, `Rule`, `select_cells`, `prior_order`, `compose`, `miou_of` (Task 5); `LatencyTables` (Task 6); `FullFrameCurve`, `delta_miou_at_equal_latency`, `latency_saved_at_equal_miou` (Task 7); the Task 3 full-frame npz; `eval.results.make_run_record`.
- Produces:
  - `python -m scripts.method2 sweep --split train [--weights results/method2/class_weights.json] [--out experiments/method2/sweep_train.jsonl]`. It writes one JSON line per (pair, scale, size, k, rule), as a `make_run_record` record whose `method` holds every pair, geometry and Rule field. `metrics` has `miou` and `per_class_iou`. `cost` has `pixel_fraction`, `gflops`, `t4_latency` (the median, p95 and parts from Task 6), `fullframe_gflops`, `flops_implied_speedup`, `realised_speedup`, `delta_miou_at_equal_latency` and `latency_saved_at_equal_miou`. The rules are:
    - `oracle`;
    - `random` (seeds 0–4, reported as one line with the mean mIoU and the min–max);
    - `prior`;
    - `all`, once per (pair, scale, size), with `t4_latency` null because k = cells isn't measured;
    - `uncertainty` × score {entropy, margin, maxprob} × pool {mean, top10} × selection {topk, greedy};
    - plus `uncertainty` with the tuned class weights when `--weights` is given.
  - `python -m scripts.method2 tune-weights --split train --pair <ctx>+<gl> --scale s --size q --score <kind> [--out results/method2/class_weights.json]`. It runs a deterministic coordinate ascent over the 19 classes in class-id order: values {0.25, 0.5, 1, 2, 4}, start at all 1, 2 passes, pool mean, topk. The objective is the mean over k ∈ {1, 2, 4, 8} of ΔmIoU at equal latency on train. It writes the weights and the objective trace, keyed by pair, scale, size and score.

- [ ] **Step 1: Write the failing tests** (fixture caches, plus `LatencyTables.from_records` with synthetic area, batch and overhead records for the fixture grid):
  - `sweep` writes the expected number of lines;
  - every line has the fields above;
  - the oracle line's mIoU is ≥ every other rule's at the same (pair, scale, size, k);
  - `tune-weights` is deterministic (two runs, identical JSON) and its objective never decreases over the trace.
- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_method2_cli.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `sweep` and `tune-weights`**
- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_method2_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/method2.py tests/test_method2_cli.py
git commit -m "Method 2 CLI: rule sweep and class-weight tuning on train"
```

### Task 10: `finalists` and `report`

**Files:**
- Modify: `scripts/method2.py`
- Test: `tests/test_method2_cli.py`

**Interfaces:**
- Consumes: the sweep JSONL (Task 9); `clip_bootstrap` (Task 7); the val caches and val full-frame npz.
- Produces:
  - `python -m scripts.method2 finalists --sweep experiments/method2/sweep_train.jsonl --out results/method2/finalists.json`. Per pair, it excludes the oracle, random and all rules, then keeps:
    - (a) the Pareto front over (T4 median latency ↓, mIoU ↑);
    - (b) for each full-frame scale of the glimpse model, the best-mIoU configuration whose predicted median latency ≤ that scale's latency.

    It removes duplicates and writes the configurations only (no val numbers).
  - `python -m scripts.method2 report --split val --finalists results/method2/finalists.json --out results/method2/val.json`. For every finalist it gives the val mIoU, per-class IoU, latency, GFLOPs, both trade-off readings and `clip_bootstrap` intervals (n = 1000, seed 0). For the reference, it adds the oracle, prior and tiling-bound rules at the finalists' geometry. It also writes a `make_run_record` JSON per finalist to `experiments/method2/val/`. It refuses `--split train`, because val is the reporting split.

- [ ] **Step 1: Write the failing tests** (fixture):
  - every finalist appears in the sweep, and none has rule kind oracle, random or all;
  - a configuration dominated in both latency and mIoU isn't on the front;
  - `report` writes the intervals and refuses `--split train`.
- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_method2_cli.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `finalists` and `report`**
- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/method2.py tests/test_method2_cli.py
git commit -m "Method 2 CLI: finalists from train, one-shot val report with bootstrap"
```

### Task 11: The runs, the fp16 check and the results README

**Files:**
- Create: `scripts/method2_plots.py`
- Create: `results/method2/README.md`, `results/method2/finalists.json`, `results/method2/val.json`, `results/method2/class_weights.json`, `results/method2/sweep_train_summary.json`
- Modify: `README.md` (the "Where we are" and Next steps sections)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `python -m scripts.method2_plots --sweep … --val results/method2/val.json --out results/method2/` writes `train_<pair>.png` and `val_<pair>.png`: mIoU vs T4 latency, with the full-frame curve, the oracle and tiling bounds, and the finalists marked.
  - `sweep_train_summary.json` holds the train Pareto points and bounds per pair. The full JSONL stays in `experiments/` because of its size.

- [ ] **Step 1: Build the train caches** (background; keep going on other steps):

```bash
python -m scripts.method2 cache --split train
```

Run it before anything else in this task, because it takes the most time.

- [ ] **Step 2: Sweep train, tune the weights for the best score kind per (pair, scale, size), sweep again with `--weights`, then compute the finalists.**
- [ ] **Step 3: Build the val caches, then run `report --split val`.** Run it once. Don't change any rule or configuration after seeing these numbers.
- [ ] **Step 4: fp16 check.** For the first finalist of each pair, rebuild its val caches with the backbones in fp16 (`--precision fp16`; add the flag to `cache` with a default of fp32). Expected: |ΔmIoU| < 0.1. Record the value.
- [ ] **Step 5: Write `results/method2/README.md`.** It covers:
  - the protocol;
  - the BiSeNetV2 checkpoint split (Task 3, Step 6);
  - the tiling-bound and oracle gaps;
  - the train curves;
  - the finalist table on val (mIoU with its interval, T4 median and p95 latency, GFLOPs, pixel fraction, realised / implied speedup, both trade-off readings with intervals);
  - the fp16 check;
  - that p95 is a sum of the parts' p95s;
  - that the latency model's overhead was measured with entropy/mean scoring.

  Then update `README.md`.
- [ ] **Step 6: Commit and push**

```bash
git add scripts/method2_plots.py results/method2 README.md
git commit -m "Method 2: train sweep, finalists and val results"
git push
```

### Task 12: Confirm the finalists' latency on the T4

**Files:**
- Modify: `scripts/latency_floor.py` (new part `final`)
- Modify: `notebooks/latency_floor_colab.ipynb`, through the generator script kept in the scratchpad (copy it into `scripts/make_latency_notebook.py` so it lives in the repo)
- Test: `tests/test_latency_floor.py`

**Interfaces:**
- Consumes: `results/method2/finalists.json`, `FoveationConfig` (Task 1).
- Produces: `--parts final --finalists results/method2/finalists.json`. It times each finalist's `FoveatedSegmenter`, with its full config (score, pool, weights, selection), as a CUDA graph. It interleaves them in the same rounds as the full-frame graphs of both models at all five scales. It writes `records["final"]`, each record carrying its prediction from `LatencyTables`. The notebook gains a `PARTS` and `FINALISTS` cell; the defaults keep today's behaviour.

- [ ] **Step 1: Write the failing test.** A QUICK-grid case list for `final` from a two-entry finalists file has 2 finalists + 10 full-frame cases, and each finalist case's `FoveationConfig` equals the file's.
- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_latency_floor.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement the `final` part and update the notebook**
- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_latency_floor.py -q`
Expected: PASS.

- [ ] **Step 5: Commit and push, then ask the user to run 3 Colab sessions** (`SESSION = "final-s1"` … `"final-s3"`, `PARTS = "final"`). They upload the JSONs to `results/method2/`.
- [ ] **Step 6: Add the measured median and p95 next to the predictions in `results/method2/README.md`.** Commit and push.
