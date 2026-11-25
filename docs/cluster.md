# Running on the school cluster (Telecom Paris, Slurm)

For any session, human or Claude, that needs to send a GPU job there. Written
2026-11-16 and checked the same day against the cluster (Method 5 chunk 3
submitted with the job script below).

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

## What is there (checked 2026-11-16)

All under the home folder (`~` is `/home/infres/<user>`, on a shared disk seen
by every node):

| Path | What |
|---|---|
| `~/avrh` | The repository (public, cloned over https: `git pull` works from the gateway). On `warp-bound` since 2026-11-16. |
| `~/avrh-venv` | Python 3.12.3 venv (`module load python/3.12.3`), torch 2.14.1 + cu126, the project installed editable with `.[reuse,dev]`. Built by `~/avrh_jobs/setup_env.sh`. |
| `~/avrh_data/vps_train_all_chunk{1,2,3}.zip` | Cityscapes-VPS `train_all`, 100 clips per zip (about 7 GB each), made on the laptop by `scripts/make_train_img_all.py --zip-chunks 4`. **Chunk 0 is not there** (it ran on the laptop). No `val/`. |
| `~/avrh_data/PIDNet_S_Cityscapes_val.pt` | PIDNet-S weights. BiSeNetV2's are in `~/.cache/torch/hub/checkpoints`. |
| `~/avrh_results/method5/<run>/` | Bound files written by the jobs (`train_all`, `train_all_ext`, `train_all_mag`). |
| `~/avrh_jobs/` | Job logs (`<job name>_<job id>.out`) and the Method 3 job scripts. |

A job extracts its zip to `/tmp/avrh_vps_<job id>` on the node and deletes it
at the end. Because that root holds one chunk only, the job runs `bound`
without `--chunk` and names the file with `--tag c<i>of4`.

`score` needs `experiments/method2/fullframe/` and the latency-floor JSONs,
which are not on the cluster: copy the bound files back and score on the
laptop (CPU, a minute).

## Submitting a Method 5 job

The job script is in the repository: [scripts/cluster/m5_bound.sh](../scripts/cluster/m5_bound.sh)
(one chunk, one model; P100, 6 CPUs, 32 GB, 14 h limit). On the gateway:

```bash
cd ~/avrh && git pull --ff-only
for M in pidnet_s bisenetv2; do
  sbatch --job-name=m5_c1_$M --output=$HOME/avrh_jobs/%x_%j.out          --export=ALL,CHUNK=1,MODEL=$M,RUN=train_all scripts/cluster/m5_bound.sh
done
squeue -u $USER
tail -n 2 ~/avrh_jobs/m5_c1_*.out
```

- `RUN` is `train_all` (registered grid), `train_all_ext` (α 0.1 and 0.15) or
  `train_all_mag` (scale 1, `--d-max 2`); the grid options are set inside the
  script.
- A finished job ends its log with `JOB_OK`. `bound` appends each finished
  clip and skips the clips already in the file, so a job that hits the time
  limit or fails midway is simply submitted again.
- Measured on 2026-11-16 (registered grid, Tesla P100): about 19 s per clip
  for PIDNet-S and 22 s for BiSeNetV2, so about 35 minutes per chunk and
  model. The QoS runs 4 jobs per user at a time; the others wait
  (`QOSMaxJobsPerUserLimit`) and start by themselves.
- The extensions need chunk 0 too: upload `vps_train_all_chunk0.zip` first
  (`scp` to `~/avrh_data/`, 7 GB, about 11 MB/s through the VPN).
- A lesson from Method 3: do not test `pytest -q` output with
  `grep " 1 passed"`; use the exit code.

### Method 5 phase 2: the static prior P

[scripts/cluster/m5_prior.sh](../scripts/cluster/m5_prior.sh) fits one chunk of the
prior for one model (P100, 2 h limit): the teacher and the uniform pass at each
stage B scale on every labelled frame, no warped pass. Same loop as above,
without `RUN`:

```bash
for C in 0 1 2 3; do for M in pidnet_s bisenetv2; do
  sbatch --job-name=m5p_c${C}_$M --output=$HOME/avrh_jobs/%x_%j.out \
      --export=ALL,CHUNK=$C,MODEL=$M scripts/cluster/m5_prior.sh
done; done
```

The chunks land in `~/avrh_results/method5/prior/`; copy them back and run
`python -m scripts.method5 prior-merge --split train_all --out-dir <that folder>`
on the laptop, which writes `results/method5/phase2/prior_<model>.pt` and `.json`.

### Method 5 phase 2, stage B: the closed loop

[scripts/cluster/m5_stream.sh](../scripts/cluster/m5_stream.sh) runs the stage B
grid of one model on one chunk, for each variant gate A kept (`r2u1` and
`r3u1`), one file per variant. It needs the committed prior and
`gate_a.json`, so `git pull` first. With P only, the runner reads the 6
labelled frames of a clip and not all 30: a fixed warp does not depend on the
frames before it, so the confusions are the same.

First the pipeline check, 5 clips of chunk 0 per model (`CHECK=1`, tag
`check5`):

```bash
for M in pidnet_s bisenetv2; do
  sbatch --job-name=m5s_check_$M --output=$HOME/avrh_jobs/%x_%j.out \
      --export=ALL,CHUNK=0,MODEL=$M,CHECK=1 scripts/cluster/m5_stream.sh
done
```

Copy `~/avrh_results/method5/stream/stream_train_all_*_check5.jsonl.gz` to a
folder of their own on the laptop, then `merge --kind stream --no-expect` and
`score2 --no-expect --out-dir <that folder> --finalists-dir <that folder>`:
every `r2u1` uniform row must be within 0.1 mIoU point of the full frame.
Then the 400 clips, the same loop over `C in 0 1 2 3` without `CHECK`, with
the time limit set to twice the check's time per clip. Copy the chunk files
to `experiments/method5/phase2/train_all/`, then `merge --kind stream` and
`score2` there.

### Method 5 phase 2, amendment of 2026-11-17: P on the scored pixels, r2 only

The prior is refitted counting only the pixels whose label is not ignored
(`SCORED=1`), then stage B runs again with it (`PRIOR=scored`), r2u1 only.
Both write to folders of their own, so the first run's files stay as they are:

```bash
cd ~/avrh && git pull --ff-only
for C in 0 1 2 3; do for M in pidnet_s bisenetv2; do
  sbatch --job-name=m5ps_c${C}_$M --output=$HOME/avrh_jobs/%x_%j.out \
      --export=ALL,CHUNK=$C,MODEL=$M,SCORED=1 scripts/cluster/m5_prior.sh
done; done
```

Copy `~/avrh_results/method5/prior_scored/*.jsonl.gz` to a folder on the
laptop, then `prior-merge --split train_all --scored-only --out-dir <that
folder>`: it writes `results/method5/phase2/prior_scored_<model>.pt` and
`.json`. Commit and push them, `git pull` on the gateway, then:

```bash
for C in 0 1 2 3; do for M in pidnet_s bisenetv2; do
  sbatch --job-name=m5ss_c${C}_$M --output=$HOME/avrh_jobs/%x_%j.out \
      --export=ALL,CHUNK=$C,MODEL=$M,PRIOR=scored,VARIANTS=r2u1 scripts/cluster/m5_stream.sh
done; done
```

Copy `~/avrh_results/method5/stream_scored/*.jsonl.gz` to
`experiments/method5/phase2/train_all_scored/`, then
`merge --kind stream --prior scored` and
`score2 --prior scored --pricing paired --rule-variants r2u1`. score2 labels
its files `scored` (`score2_train_all_<model>_scored.json` and
`gate_b_train_all_scored.json` in that folder, `finalists_<model>_scored.json`
in `results/method5/phase2/`).

## Bringing results back

```bash
scp -i ~/.ssh/id_ed25519_cluster "<user>@gpu-gw.enst.fr:avrh_results/method5/train_all/bound_*_c[123]of4.jsonl.gz" experiments/method5/train_all/
```

Then `merge` and `score` on the laptop, and log the run in the method's
TRIES.md (job id, GPU, commit, time per clip).

## Method 11: fine-tuning on warped frames

Spec: [2026-11-25-method11-warp-finetune-design.md](superpowers/specs/2026-11-25-method11-warp-finetune-design.md).
Scripts: [m11_train.sh](../scripts/cluster/m11_train.sh) (one run, P100, 24 h limit, resumable per
epoch) and [m11_eval.sh](../scripts/cluster/m11_eval.sh) (one run's confusions on one chunk of
`train_all`, or the public weights at the 13 scales).

**Before the first job** (on the laptop):

1. `python scripts/get_val.py --out <cityscapes folder> --train` fetches `leftImg8bit/train` and
   `gtFine/train` (about 6 GB) with the user's Cityscapes login.
2. `python -m scripts.method11 boundary-prior --cityscapes <cityscapes folder>` writes
   `results/method11/prior_boundary.pt` and `.json` (CPU, minutes); commit them, `git pull` on the cluster.
3. Zip the two folders as `cityscapes_train.zip` (paths inside: `leftImg8bit/train/...`,
   `gtFine/train/...`) and `scp` it to `~/avrh_data/`.
4. Gate 1 scores all 400 clips, so `vps_train_all_chunk0.zip` must be in `~/avrh_data/` too.

**Gate 0** (spec): per model, `EXTRA="--lr 0 --max-steps 20"` at `KIND=uniform SCALE=1` with
`TAG=gate0_lr0`, then `EXTRA="--epochs 1"` at `KIND=uniform SCALE=0.5` with `TAG=gate0_ep1`;
evaluate both on 5 clips of a chunk.

**Gate 1**, per model (20 runs in all; the QoS runs 4 at a time):

```bash
for M in bisenetv2 pidnet_s; do
  for S in 0.25 0.5 0.75 1; do for K in uniform boundary; do
    sbatch --job-name=m11_${M}_${K}_c$S --output=$HOME/avrh_jobs/%x_%j.out \
        --export=ALL,MODEL=$M,KIND=$K,SCALE=$S scripts/cluster/m11_train.sh
  done; done
  sbatch --job-name=m11_${M}_multi --output=$HOME/avrh_jobs/%x_%j.out \
      --export=ALL,MODEL=$M,KIND=multi scripts/cluster/m11_train.sh
  sbatch --job-name=m11_${M}_error_c0.25 --output=$HOME/avrh_jobs/%x_%j.out \
      --export=ALL,MODEL=$M,KIND=error,SCALE=0.25 scripts/cluster/m11_train.sh
done
```

Then `m11_eval.sh` for each finished run folder and `PUBLIC=<model>`, chunks 0 to 3.
