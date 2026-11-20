#!/bin/bash
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=14:00:00
# One chunk of the Methods 7/8 run (dense full frame + guided upsampling grid) for one model, on the
# school cluster. Spec: docs/superpowers/specs/2026-11-20-method7-8-guided-upsampling-scale-oracle-design.md
#
#   cd ~/avrh && sbatch --job-name=m7_c0_pidnet_s --output=$HOME/avrh_jobs/%x_%j.out \
#       --export=ALL,CHUNK=0,MODEL=pidnet_s scripts/cluster/m7_run.sh
#
# CHUNK: 0..3, read from ~/avrh_data/vps_train_all_chunk$CHUNK.zip (the 100 clips of that chunk).
# MODEL: pidnet_s | bisenetv2. EXTRA: optional extra arguments (e.g. the finalists' objects run:
#   EXTRA="--ff-scales none --configs <names> --obj all --tag fin_c0of4").
# Output: ~/avrh_results/method7/train_all/run_train_all_${MODEL}_c${CHUNK}of4.jsonl.gz, appended
# clip by clip: a job that hits the time limit is submitted again and resumes.
set -euo pipefail
: "${CHUNK:?}" "${MODEL:?}"
EXTRA=${EXTRA:-"--obj ff --tag c${CHUNK}of4"}

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
python -m pytest -q tests/test_guided_upsample.py tests/test_scale_oracle.py

# The root holds only this chunk's clips, so no --chunk; the tag names the file.
OUT=$HOME/avrh_results/method7/train_all
mkdir -p "$OUT"
# shellcheck disable=SC2086
python -u -m scripts.method7 run --split train_all --models "$MODEL" --out-dir "$OUT" $EXTRA
echo JOB_OK
