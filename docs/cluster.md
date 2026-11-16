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
  model. Several jobs ran at the same time without trouble.
- The extensions need chunk 0 too: upload `vps_train_all_chunk0.zip` first
  (`scp` to `~/avrh_data/`, 7 GB, about 11 MB/s through the VPN).
- A lesson from Method 3: do not test `pytest -q` output with
  `grep " 1 passed"`; use the exit code.

## Bringing results back

```bash
scp -i ~/.ssh/id_ed25519_cluster "<user>@gpu-gw.enst.fr:avrh_results/method5/train_all/bound_*_c[123]of4.jsonl.gz" experiments/method5/train_all/
```

Then `merge` and `score` on the laptop, and log the run in the method's
TRIES.md (job id, GPU, commit, time per clip).
