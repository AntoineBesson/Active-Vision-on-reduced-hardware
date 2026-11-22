#!/bin/bash
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4:00:00
# One chunk of the Method 9 features run (every frame of each clip at scale 0.5, PIDNet-S), on the
# school cluster. Spec: docs/superpowers/specs/2026-11-22-method9-scale-selector-design.md
#
#   cd ~/avrh && sbatch --job-name=m9_c0 --output=$HOME/avrh_jobs/%x_%j.out \
#       --export=ALL,CHUNK=0 scripts/cluster/m9_features.sh
#
# CHUNK: 0..3, read from ~/avrh_data/vps_train_all_chunk$CHUNK.zip (the 100 clips of that chunk).
# Output: ~/avrh_results/method9/train_all/features_train_all_pidnet_s_c${CHUNK}of4.jsonl.gz, appended
# clip by clip: a job that hits the time limit is submitted again and resumes.
set -euo pipefail
: "${CHUNK:?}"

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
python -m pytest -q tests/test_method9.py

# The root holds only this chunk's clips, so no --chunk; the tag names the file.
OUT=$HOME/avrh_results/method9/train_all
mkdir -p "$OUT"
python -u -m scripts.method9 features --split train_all --out-dir "$OUT" --tag "c${CHUNK}of4"
echo JOB_OK
