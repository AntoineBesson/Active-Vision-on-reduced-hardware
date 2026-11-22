# Method 9: a realisable per-frame scale selector (PIDNet-S)

Status: **accepted 2026-11-22 (Antoine chose both selectors, S1 and S2). Nothing has run.**
Frozen at its first run; changes go in a dated Amendments section.

## Why, and the target

Method 8's oracle bound passed (train_all, PIDNet-S +4.85 vs the envelope at the ff 0.5 budget), and
its predictability screen passed narrowly (2026-11-22 amendment of the Method 7/8 spec): the stale
oracle keeps **+1.08 [+0.41, +1.81]** at the ff 0.5 budget (7.04 ms), and nothing else clears +1.0.
BiSeNetV2's stale oracle is negative at every budget, so this method is **PIDNet-S only**, at the
**ff 0.5 budget** (0.625 is reported, not ruled on). Shuffling the oracle's choices inside a clip
does as well as the stale oracle, and one scale per clip keeps +3.09: what is predictable is mostly
which clip needs which scales. The selector therefore reads running statistics of the clip's own
past outputs. Realistic target: well under +1 point.

## The selector

At frame t, the scale is chosen from the 11 dense scales using only frames before t:

* **Features** (from each past frame's own logits, on the logit grid, no extra pass): mean pixel
  entropy, mean max-probability, the 19-class pixel fractions of the prediction, and the fraction of
  logit-grid cells on a class boundary. Kept as two exponential moving averages over the clip's past
  frames (half-lives 2 and 10 frames); the first frame of a clip takes the budget scale.
* **S1, threshold (nothing fitted but cut points):** the short-EMA entropy, mapped to scales by
  cut points set on train so the scale shares equal the oracle's at that budget, then the cut points
  shifted together until the train mean latency meets the budget.
* **S2, small fitted rule:** ridge regression from the 44 features to the oracle's chosen scale
  index; frames ranked by the prediction and given scales with the same share-matching. Cross-fitted
  on train by 5 folds of clips (each clip predicted by a fit that never saw it); the val run uses
  one fit on all of train_all. The segmentation model is frozen; only these 45 numbers are fitted.

Both selectors run in **closed loop** on val (each frame at the scale chosen, the next frame's
features from that output). On train, features come from an open-loop pass of every frame at the
budget scale (0.5): that is what the run below records, and it is stated as an approximation.

## Cost

Mean latency = mean over frames of the full frame's T4 price at the chosen scale + the selector's
overhead (features from the logits, a dot product, a table lookup), timed on the T4 with Method 7's
stage 2 sessions. Switching scales between frames needs one CUDA graph per scale; the timing
session checks that replaying different graphs back to back costs the same as replaying one.

## Gates and pass rule

1. **Train gate (train_all, cross-fitted, zero selector overhead):** a selector continues if its Δ
   mIoU vs the envelope at its mean latency is **≥ +0.5** at the ff 0.5 budget. Neither: Method 9
   stops, negative, and Method 8 is closed.
2. **T4 timing** of the survivors' overhead.
3. **Val, read once, closed loop:** pass if Δ vs the envelope at the measured mean latency is
   **above 0 with its 95% clip-bootstrap interval above 0**. Moving-object recall (train_all,
   open-loop) is reported against the same model's full-frame curve.

## Runs

* `features` (GPU, new): every frame of every train_all clip at scale 0.5, the per-frame features
  saved (about 6 min per chunk on a P100). The labelled frames' confusions at every scale already
  exist in the Method 7 run files.
* `select` (CPU): S1 and S2 on train, cross-fitted, gate 1.
* val: closed loop on the 50 val clips, all frames, PIDNet-S (laptop or cluster, short).

```bash
# train_all features (cluster, 4 chunks): scripts/cluster/m9_features.sh
python -m scripts.method9 merge --split train_all --out-dir ~/avrh_results/method9/train_all
python -m scripts.method9 select --split train_all --out-dir ~/avrh_results/method9/train_all \
    --run-dir ~/avrh_results/method7/train_all                     # gate, writes results/method9/selectors_pidnet_s.json
# val (after gate and T4 timing): dense full frame for the envelope, then the closed loop
python -m scripts.method7 run --split val --models pidnet_s,bisenetv2 --guided-scales none
python -m scripts.method7 merge --split val
python -m scripts.method9 closedloop --split val --kinds <survivors>
python -m scripts.method9 score --split val --overhead-ms <T4 overhead>
```
