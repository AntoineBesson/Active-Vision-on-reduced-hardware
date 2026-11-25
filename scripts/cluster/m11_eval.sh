#!/bin/bash
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# Method 11: the per-clip confusions of one finished run (or of the public weights) on one chunk
# of VPS train_all (docs/cluster.md; spec docs/superpowers/specs/2026-11-25-method11-warp-finetune-design.md).
#
#   # a finished run
#   sbatch --job-name=m11e_c1_bisenetv2_B_c0.25 --output=$HOME/avrh_jobs/%x_%j.out \
#       --export=ALL,CHUNK=1,RUN=bisenetv2_boundary_c0.25 scripts/cluster/m11_eval.sh
#   # the frozen public weights at the 13 scales (the envelope's frozen curves)
#   sbatch --job-name=m11e_c1_public_bisenetv2 --output=$HOME/avrh_jobs/%x_%j.out \
#       --export=ALL,CHUNK=1,PUBLIC=bisenetv2 scripts/cluster/m11_eval.sh
#
# CHUNK: 0..3, from ~/avrh_data/vps_train_all_chunk$CHUNK.zip. RUN: a folder name under
# ~/avrh_results/method11/runs. Output: ~/avrh_results/method11/eval/train_all/, appended clip by
# clip (resumable).
set -euo pipefail
: "${CHUNK:?}"
OUT=$HOME/avrh_results/method11/eval/train_all
if [ -n "${PUBLIC:-}" ]; then
  WHAT=(--public --models "$PUBLIC" --kind multi)
else
  : "${RUN:?}"
  WHAT=(--run-dir "$HOME/avrh_results/method11/runs/$RUN")
fi

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

mkdir -p "$OUT"
python -u -m scripts.method11 eval "${WHAT[@]}" --split train_all --tag "c${CHUNK}of4" --out-dir "$OUT"
echo JOB_OK
