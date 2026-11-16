#!/bin/bash
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=14:00:00
# One chunk of the Method 5 bound for one model, on the school cluster (docs/cluster.md).
#
#   cd ~/avrh && sbatch --job-name=m5_c1_pidnet_s --output=$HOME/avrh_jobs/%x_%j.out \
#       --export=ALL,CHUNK=1,MODEL=pidnet_s,RUN=train_all scripts/cluster/m5_bound.sh
#
# CHUNK: 0..3, read from ~/avrh_data/vps_train_all_chunk$CHUNK.zip (the 100 clips of that chunk).
# MODEL: pidnet_s | bisenetv2.
# RUN:   train_all (registered grid) | train_all_ext (alpha 0.1, 0.15) | train_all_mag (scale 1,
#        d_max 2). The grid options live here because --export cannot carry commas.
# Output: ~/avrh_results/method5/$RUN/bound_train_all_${MODEL}_c${CHUNK}of4.jsonl.gz, appended clip
# by clip: a job that hits the time limit is submitted again and resumes.
set -euo pipefail
: "${CHUNK:?}" "${MODEL:?}"
RUN=${RUN:-train_all}
case "$RUN" in
    train_all)     EXTRA="" ;;
    train_all_ext) EXTRA="--alphas 0.1,0.15" ;;
    train_all_mag) EXTRA="--scales 1 --d-max 2 --saliencies B,S --alphas 0.1,0.25,0.5" ;;
    *) echo "unknown RUN=$RUN"; exit 2 ;;
esac

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
python -m pytest -q tests/test_warp_grid.py

# The root holds only this chunk's clips, so no --chunk (it would cut the 100 clips in four):
# the tag alone names the file like a --chunk run on the full dataset does.
OUT=$HOME/avrh_results/method5/$RUN
mkdir -p "$OUT"
python -u -m scripts.method5 bound --split train_all --models "$MODEL" --tag "c${CHUNK}of4" --out-dir "$OUT" $EXTRA
echo JOB_OK
