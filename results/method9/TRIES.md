# Method 9 (reuse the slow branch across frames): tries

Spec: [docs/superpowers/specs/2026-11-23-method9-deep-feature-reuse-design.md](../../docs/superpowers/specs/2026-11-23-method9-deep-feature-reuse-design.md).

| Date | What | Where | Outcome |
|---|---|---|---|
| 2026-11-23 | Spec drafted (gate 0 cost split, gate 1 staleness bound) | cloud thread | nothing coded or run; awaits Antoine's acceptance |
| 2026-11-24 | Spec accepted (scales 0.5 and 1, +1.0 margins, moving objects in gate 1, gate 1 on the cluster); written as frozen but NOT model-agnostic; literature check: mechanism = Clockwork Convnets 2016 | cloud thread | gate 0 coded; FLOP prediction: BiSeNetV2 fresh = 95 % of full (expected fail), PIDNet-S 67 % (ceiling ~+1 at s=1, several points at 0.5) |
