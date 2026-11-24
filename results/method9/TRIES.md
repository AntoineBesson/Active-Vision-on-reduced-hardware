# Method 9 (reuse the slow branch across frames): tries

Spec: [docs/superpowers/specs/2026-11-23-method9-deep-feature-reuse-design.md](../../docs/superpowers/specs/2026-11-23-method9-deep-feature-reuse-design.md).

| Date | What | Where | Outcome |
|---|---|---|---|
| 2026-11-23 | Spec drafted (gate 0 cost split, gate 1 staleness bound) | cloud thread | nothing coded or run; awaits Antoine's acceptance |
| 2026-11-24 | Spec accepted (scales 0.5 and 1, +1.0 margins, moving objects in gate 1, gate 1 on the cluster); written as frozen but NOT model-agnostic; literature check: mechanism = Clockwork Convnets 2016 | cloud thread | gate 0 coded; FLOP prediction: BiSeNetV2 fresh = 95 % of full (expected fail), PIDNet-S 67 % (ceiling ~+1 at s=1, several points at 0.5) |
| 2026-11-24 | Gate 0 on a Colab T4 (1 session, envelope priced in the same session) | Colab | **PASSES at s = 0.5 for both models, FAILS at s = 1 for both.** PIDNet-S 0.5: fresh 4.70 ms = 74 % of key 6.34 ms, ceiling +4.72 (K=inf), +2.10/+2.97/+3.40/+3.66 at K=2..5. BiSeNetV2 0.5: fresh 6.90 = 76 % of 9.02 (its semantic branch is 2.2 ms despite 5 % of FLOPs), ceiling +1.77 (K=inf), +0.08/+0.82/+1.06/+1.20. s = 1: PIDNet-S +0.41 max, BiSeNetV2 -0.22. Gate 1 runs s = 0.5 only. [gate0_report.json](gate0_report.json) |
