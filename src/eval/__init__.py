"""Evaluation: metrics, compute cost and run records.

* :mod:`eval.metrics` -- confusion-matrix mIoU / per-class IoU accumulated
  over the labelled frames of all clips (numpy).
* :mod:`eval.cost` -- per-component cost ledger, FLOPs (torch's built-in
  FlopCounterMode), GPU latency (median/p95) and peak memory.
* :mod:`eval.results` -- one JSON file per run (config, git, seed, metrics, cost).
* :mod:`eval.harness` -- generic evaluation loop and the full-frame method (torch).

``eval``, ``eval.metrics``, ``eval.cost`` and ``eval.results`` import without torch.
"""
