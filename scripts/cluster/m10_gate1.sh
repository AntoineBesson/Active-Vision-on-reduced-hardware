#!/bin/bash
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=8:00:00
# One chunk of the Method 10 gate 1 run (staleness of the slow branch, scale 0.5) for one model,
# on the school cluster. Spec: docs/superpowers/specs/2026-11-23-method10-slow-branch-reuse-design.md
#
#   cd ~/avrh && git fetch origin claude/project-thread-pbd01q && git checkout claude/project-thread-pbd01q
#   for C in 0 1 2 3; do for M in pidnet_s bisenetv2; do
#     sbatch --job-name=m10_c${C}_$M --output=$HOME/avrh_jobs/%x_%j.out --export=ALL,CHUNK=$C,MODEL=$M scripts/cluster/m10_gate1.sh
#   done; done
#
# CHUNK: 0..3, read from ~/avrh_data/vps_train_all_chunk$CHUNK.zip (the 100 clips of that chunk).
# MODEL: pidnet_s | bisenetv2.
# Output: ~/avrh_results/method10/train_all/run_train_all_${MODEL}_c${CHUNK}of4.jsonl.gz, appended
# clip by clip: a job that hits the time limit is submitted again and resumes.
set -euo pipefail
: "${CHUNK:?}" "${MODEL:?}"

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
python -m pytest -q tests/test_method10_split.py tests/test_method10_gate1.py

# The root holds only this chunk's clips, so no --chunk; the tag names the file.
OUT=$HOME/avrh_results/method10/train_all
mkdir -p "$OUT"
python -u -m scripts.method10_gate1 run --split train_all --models "$MODEL" --obj --tag "c${CHUNK}of4" --out-dir "$OUT"
echo JOB_OK
