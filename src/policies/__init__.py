"""Gaze / foveation policies: which regions of a frame to inspect.

* :mod:`policies.interfaces` -- the ``GazePolicy`` contract, ``GazeState`` and
  the normalised ``Box`` convention. Memory-free (e.g. uncertainty-driven) and
  canvas-aware policies share the same contract. Imports without torch.
"""
