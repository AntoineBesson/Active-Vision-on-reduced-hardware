# Running on the school cluster (Telecom Paris, Slurm)

For any session, human or Claude, that needs to send a GPU job there. Written
2026-11-16 from the notes of the Method 3 runs (2026-11-05); the cluster was
not reachable when this was written, so **the lines marked "to confirm" must
be checked at the next connection and this file corrected**.

## What it is for

- **Accuracy runs only** (caches, sweeps, bounds). Its GPUs are Tesla P100s:
  no latency measured there enters a result. T4 latency stays on Colab.
- Method 3 chunk 3 ran there; the same clip on the laptop and on a P100 gave
  the same numbers (results/method3/TRIES.md, 2026-11-05).

## Connecting

1. **VPN first.** Outside the school network the gateway does not answer
   (SSH times out; seen on 2026-11-10, 12 and 16). The Telecom Paris OpenVPN
   asks for a password at each connection: **the user types it**. No session
   reads or stores it, and the password file under `env/` is never opened by
   Claude.
2. **SSH with the key made for these sessions** (no passphrase), on the
   laptop:

   ```bash
   ssh -i ~/.ssh/id_ed25519_cluster -o IdentitiesOnly=yes <user>@gpu-gw.enst.fr
   ```

   `<user>` is the school login. It is kept out of the repository; a Claude
   session on the laptop finds it in its project memory
   (`reference-telecom-cluster`). A cloud session has neither the key nor the
   VPN and cannot reach the cluster: it must hand the job to a laptop session
   or to the user.
3. A quick test that does nothing: `ssh ... "sinfo -s"`.

## Rules of the place

- **Never compute on the gateway** (`gpu-gw`): only `sbatch`, `squeue`,
  `scancel`, file copies.
- Partitions: `P100` (default, 16 GB Tesla P100, up to 2 days), `3090` (often
  full), `CPU`, `cpu-high`. QoS `qos-student`: 36 h of wall time at most.
- No conda on the gateway. `module avail` lists Python 3.10 to 3.13; the
  project runs in a venv built from one of them.
- Node local scratch: `/tmp` (845 GB free on node16 on 2026-11-05).
- Job scripts live in `~/avrh_jobs` on the cluster, not in this repository.

## What must be there before a job (to confirm)

- The repository at the commit to run (`git clone` / `git pull`, branch
  `warp-bound` for Method 5).
- A venv with the project's requirements and a CUDA build of torch that
  supports the P100.
- Cityscapes-VPS with **every frame** (`train_all`), and the PIDNet-S
  weights; `CITYSCAPES_VPS_ROOT` and `PIDNET_S_WEIGHTS` set in the job. The
  data left by the Method 3 runs has no `val/` folder, which only matters for
  tests that read val.
- `experiments/method2/fullframe/` and the latency-floor JSONs if `score` is
  run there. It is simpler to copy the bound files back and score on the
  laptop (CPU, a minute).

Paths on the cluster (repository, venv, data) are **not recorded here yet**:
write them in this section at the next connection.

## A job, as a template (untested as written)

```bash
#!/bin/bash
#SBATCH --job-name=m5-c1
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd <repo on the cluster>
source <venv>/bin/activate
export CITYSCAPES_VPS_ROOT=<data root> PIDNET_S_WEIGHTS=<weights file> PYTHONUNBUFFERED=1
python -m pytest -q tests/test_warp_grid.py            # a quick check that the install works
python -m scripts.method5 bound --split train_all --chunk 1/4
```

- `bound` appends each finished clip and skips the clips already in the file,
  so a job that hits the time limit is simply submitted again.
- One job per chunk and per run folder; the extensions use their own
  `--out-dir` (commands in results/method5/README.md, "How to run").
- A lesson from Method 3: do not test `pytest -q` output with
  `grep " 1 passed"`; use the exit code.

## Bringing results back

```bash
scp -i ~/.ssh/id_ed25519_cluster "<user>@gpu-gw.enst.fr:<repo>/experiments/method5/train_all/bound_*_c1of4.jsonl.gz" experiments/method5/train_all/
```

Then `merge` and `score` on the laptop, and log the run in the method's
TRIES.md (job id, GPU, commit, time per clip).
