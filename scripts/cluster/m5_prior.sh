#!/bin/bash
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# One chunk of the Method 5 phase 2 static prior P for one model, on the school cluster
# (docs/cluster.md): per labelled frame, the teacher and the uniform pass at each stage B scale
# of the model, the E pixels counted per cell, and the density rounds its saliencies need.
#
#   cd ~/avrh && sbatch --job-name=m5p_c1_pidnet_s --output=$HOME/avrh_jobs/%x_%j.out \
#       --export=ALL,CHUNK=1,MODEL=pidnet_s scripts/cluster/m5_prior.sh
#
# CHUNK: 0..3, read from ~/avrh_data/vps_train_all_chunk$CHUNK.zip (the 100 clips of that chunk).
# MODEL: pidnet_s (scales 0.25, 0.375, 0.5) | bisenetv2 (0.25, 0.375).
# Output: ~/avrh_results/method5/prior/prior_train_all_${MODEL}_c${CHUNK}of4.jsonl.gz, appended clip
# by clip: a job that hits the time limit is submitted again and resumes. Then, from the copied
# chunks: python -m scripts.method5 prior-merge --split train_all --out-dir <folder of the chunks>
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
python -m pytest -q tests/test_warp_grid.py tests/test_warp_policy.py

# The root holds only this chunk's clips, so no --chunk; the tag names the file like a --chunk run.
OUT=$HOME/avrh_results/method5/prior
mkdir -p "$OUT"
python -u -m scripts.method5 prior --split train_all --models "$MODEL" --tag "c${CHUNK}of4" --out-dir "$OUT"
echo JOB_OK
