# Warping with frozen weights: next research points

Written 2026-11-16, after chunk 0 of the Method 5 bound (100 of 400 clips,
partial; see [results/method5/TRIES.md](../../results/method5/TRIES.md)).
Nothing here is run or decided. Every point keeps the backbone frozen; ideas
that train the backbone are listed apart at the end.

## Where the current warp stands

Method 5 resamples the frame to a smaller image with a **separable** warp: one
density for the rows, one for the columns
(`d = (1 − α) + α · p / mean(p)`, clipped to `[0.5, 1/c]`, mean 1), where `p`
is the saliency averaged over the other axis. To enlarge one object it
enlarges the whole band of rows and the whole band of columns it sits in, so
most of the extra pixels go to regions that did not ask for them, and two
salient objects in different corners also magnify the two empty corners.

What chunk 0 shows (not a reading of the rule):

- The gentlest warp of the grid (α = 0.25) is the best everywhere and stronger
  warps lose the gain. A frozen model has never seen a distorted image.
- The gain is +1 to +2 mIoU points below scale 0.75, worth 0.2 to 1.2 ms of
  full-frame latency, because the full-frame curve is steep there.
- At scale 1 the warp is the identity by construction (upper clip `1/c`).

So two things limit it: how much distortion the frozen model tolerates, and
how well the warp spends the pixels it moves. The points below are about the
second, which was assumed (wrongly) to be already local.

## 1. A local, non-separable warp (first, before any learned policy)

Make the magnification follow the saliency in 2-D, so only the neighbourhood
of a salient region grows and the rest of its rows and columns do not.

- **Kernel sampling ("Learning to Zoom").** Each output pixel reads the frame
  at the saliency-weighted mean of its neighbours' positions, with a Gaussian
  kernel. The size of the kernel sets how local the zoom is. The published
  layer is trained end to end; here only the sampling formula is taken and
  the saliency stays an oracle (then a cheap proxy).
- **Same oracle protocol as Method 5.** Same saliencies (E, B, S), same clips,
  same pass rule, same pricing; the separable warp of Method 5 is the
  baseline to beat at equal scale. This isolates the shape of the warp from
  everything else, which is what a learned policy would need to know first.
- **What must be solved.** (a) The inverse map is no longer two 1-D tables: it
  needs a 2-D inverse (a piecewise bilinear approximation of the forward map
  is the published way, in "Learning to Zoom and Unzoom"). (b) The antialiased
  resampling used now is separable; a 2-D warp needs a filter whose width
  follows the local Jacobian, or the uniform row will no longer reproduce the
  baseline (the pipeline check). (c) Border behaviour: the published warps can
  crop the image; FOVEA adds an anti-cropping constraint.
- **Distortion measure.** Report, per configuration, how far the warp is from
  a similarity (anisotropy and local scale range), to test the hypothesis that
  the frozen model fails on distortion and not on magnification.

## 2. A fixed geometric prior, no saliency per frame

In driving scenes small objects sit around the vanishing point. A warp that
always magnifies that region costs nothing to choose.

- **Static prior.** The mean saliency of the train clips (S or B) used as one
  fixed map for every frame: a lookup table, no policy, no memory. On an
  embedded card this is the cheapest possible version, and it gives the floor
  that any per-frame policy must beat.
- **Two-plane perspective prior.** A warp parameterised by the vanishing point
  and two planes (ground and a plane above it), published for detection. Here
  it would be fitted, not learned through the backbone.
- **Reading.** Same bound, with the prior in place of the oracle. If the fixed
  prior keeps most of the oracle's gain, the per-frame policy (and its
  latency) is not needed.

## 3. Magnification beyond native resolution at scale 1

Added to the Method 5 bound as an extension on 2026-11-16 (`--d-max 2` with
`--scales 1`): small distant objects are enlarged up to 2 times and the rest
is squeezed, at the full frame's own cost. It is the only version that can
move the top of the curve, where a T4 actually runs. If it gains, points 1 and
2 should be read at scale 1 first.

## 4. Less distortion for the same magnification

- **Weaker and smoother warps:** α below 0.25 (extension of 2026-11-16), and a
  smoother saliency, to find where the gain peaks.
- **Isotropic local zoom:** constrain the 2-D warp to scale both axes equally
  where it magnifies, so objects keep their aspect ratio.
- **One or two rectangular zoom regions** with smooth borders instead of a
  free-form density: closer to the crops of Method 2, but in one pass.

## 5. From oracle to a real saliency (after 1 to 4)

Only if an oracle version passes, and inside the latency budget it leaves:

- the previous frame's prediction (the bound's `prev` timing already shows it
  keeps most of the gain), then a persistent map with ages as in Method 4;
- a low-resolution pass of the same frame is not affordable in one pass, so
  the saliency must come from memory or from the fixed prior;
- a learned policy (RL or supervised on the oracle) only once the shape of
  the warp is settled by point 1.

## Out of scope here: training the backbone

Fine-tuning the segmenter on warped images is what the published zoom methods
do, and it is probably where most of their gain comes from. It breaks the
frozen-model rule of the harness, so it stays an opening for the end of the
study.

## Related work to read before building point 1

Checked by a web search on 2026-11-16 (titles, venues and abstracts only; the
numbers have not been checked against the papers):

- Recasens et al., "Learning to Zoom: a Saliency-Based Sampling Layer for
  Neural Networks", ECCV 2018, arXiv 1809.03355. The kernel sampling layer.
- Thavamani et al., "FOVEA: Foveated Image Magnification for Autonomous
  Navigation", ICCV 2021, arXiv 2108.12102. Saliency from dataset priors or
  from the previous frame's detections, anti-cropping; detection.
- Thavamani et al., "Learning to Zoom and Unzoom", CVPR 2023, arXiv
  2303.15390. Zoom the input, unzoom the features with a piecewise bilinear
  inverse; includes semantic segmentation on Cityscapes. **The closest prior
  work to Method 5**: the difference to state clearly is that the backbone is
  frozen here and the comparison is at equal measured latency.
- Ghosh et al., "Learned Two-Plane Perspective Prior based Image Resampling
  for Efficient Object Detection", CVPR 2023, arXiv 2303.14311. The geometric
  prior of point 2.
