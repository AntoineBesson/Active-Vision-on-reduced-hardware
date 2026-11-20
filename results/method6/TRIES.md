# Method 6 (local 2-D warp bound): tries

Spec: [docs/superpowers/specs/2026-11-17-method6-local-warp-bound-design.md](../../docs/superpowers/specs/2026-11-17-method6-local-warp-bound-design.md).

| Date | What | Where | Outcome |
|---|---|---|---|
| 2026-11-17 | Spec drafted, code and CPU tests written (`src/warp/grid2d.py`, `src/warp/bound2d.py`, `scripts/method6.py`) | cloud thread | no run yet; spec awaits the user's acceptance |
| 2026-11-17 | Spec accepted by the user with scale 0.75 kept (152 passes per labelled frame); runs on the school cluster | cloud thread | no run yet |
| 2026-11-17 | Smoke job: chunk 0, PIDNet-S, school cluster P100 (job 1109278), full grid (152 rows) | cluster | JOB_OK in 1 h 56 min, 69 s/clip; 71 CPU tests pass on the node |
| 2026-11-19 | First look, chunk 0, PIDNet-S, scored on the laptop CPU (100 clips; rule not read) | laptop | pipeline check ok at 4 scales (gaps ~1e-7). Script's reproduction check "NOT ok" because it compares 100 clips with Method 5's 400-clip mIoU; on the same 100 clips, against Method 5's chunk-0 bound files, the 4 separable rows match within 0.001 pt (identical confusions on all clips at 0.375, 0.5, 0.75; on 29 of 100 at 0.25). Best 2-D vs separable on these clips: +0.16, +0.11, +0.09, −0.10 (0.25 → 0.75), already far from +0.5 |
| 2026-11-19 | The other 7 jobs (chunks 1-3 PIDNet-S, chunks 0-3 BiSeNetV2; jobs 1112158-1112164) | cluster | all COMPLETED, JOB_OK, 1 h 59 to 2 h 05 each |
| 2026-11-20 | Merge and score, 400 clips, both models | laptop CPU | both checks ok at 4 scales. **Rule FAILS for both models**: best 2-D vs full frame / vs separable bar, PIDNet-S +1.01 / −0.24 (0.25), +0.62 / −0.30, +0.53 / −0.46, +0.11 / −0.10; BiSeNetV2 +1.19 / −0.62 (0.25), +0.78 / −0.39, +0.31 / −0.03, +0.04 / −0.03. The 2-D warp is below the separable one in all 8 cells; α = 0.5 hurts (mean −0.3 to −2.8 pts). The chunk-0 lead over the separable warp at 0.25 (+0.16) was noise (−0.24 on 400 clips). Written up in README.md; method stopped, next step is the user's call |
