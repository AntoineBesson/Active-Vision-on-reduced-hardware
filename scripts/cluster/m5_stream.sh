#!/bin/bash
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# One chunk of Method 5 phase 2, stage B, for one model, on the school cluster (docs/cluster.md):
# every configuration of the stage B grid (P only after gate A, plus the uniform rows) in its own
# closed loop, for each warp variant gate A kept, one file per variant.
#
#   cd ~/avrh && sbatch --job-name=m5s_c1_pidnet_s --output=$HOME/avrh_jobs/%x_%j.out \
#       --export=ALL,CHUNK=1,MODEL=pidnet_s scripts/cluster/m5_stream.sh
#
# CHUNK: 0..3, read from ~/avrh_data/vps_train_all_chunk$CHUNK.zip (the 100 clips of that chunk).
# MODEL: pidnet_s | bisenetv2. VARIANTS: default "r2u1 r3u1" (gate_a.json, 2026-11-17).
# CHECK=1: the pipeline check only, the first 5 clips of the chunk (tag check5).
# Needs results/method5/phase2/prior_$MODEL.pt and gate_a.json in the checkout (git pull first).
# Output: ~/avrh_results/method5/stream/stream_train_all_${MODEL}_<variant>_c${CHUNK}of4.jsonl.gz,
# appended clip by clip: a job that hits the time limit is submitted again and resumes.
# The 2 h limit is a first guess (P only runs the 6 labelled frames of a clip); set it to twice
# the pipeline check's measured time per clip once that is known.
set -euo pipefail
: "${CHUNK:?}" "${MODEL:?}"
VARIANTS=${VARIANTS:-"r2u1 r3u1"}

module purge
module load python/3.12.3
unset PYTHONPATH
source ~/avrh-venv/bin/activate

VPS=/tmp/avrh_vps_$SLURM_JOB_ID
trap 'rm -rf "$VPS"' EXIT
mkdir -p "$VPS"
python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$HOME/avrh_data/vps_train_all_chunk$CHUNK.zip" "$VPS"
export CITYSCAPES_VPS_ROOT=$VPS PIDNET_S_WEIGHTS=$HOME/avrh_data/PIDNet_S_Cityscapes_val.pt PYTHONUNBUFFERED=1

cd ~/avrh
git log --oneline -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -m pytest -q tests/test_warp_grid.py tests/test_warp_policy.py tests/test_warp_stream.py
test -f "results/method5/phase2/prior_$MODEL.pt" || { echo "no prior for $MODEL: git pull"; exit 1; }

# The root holds only this chunk's clips, so no --chunk for the full run; the tag names the file
# like a --chunk run. The check takes the chunk's first 5 clips (0/20 of 100).
if [ "${CHECK:-0}" = 1 ]; then SEL=(--chunk 0/20 --tag check5); else SEL=(--tag "c${CHUNK}of4"); fi
OUT=$HOME/avrh_results/method5/stream
mkdir -p "$OUT"
for V in $VARIANTS; do
  python -u -m scripts.method5 stream --split train_all --models "$MODEL" --variant "$V" "${SEL[@]}" --out-dir "$OUT"
done
echo JOB_OK
