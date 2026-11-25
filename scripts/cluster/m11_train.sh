#!/bin/bash
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# One Method 11 fine-tuning run on the school cluster (docs/cluster.md; spec
# docs/superpowers/specs/2026-11-25-method11-warp-finetune-design.md).
#
#   cd ~/avrh && sbatch --job-name=m11_bisenetv2_B_c0.25 --output=$HOME/avrh_jobs/%x_%j.out \
#       --export=ALL,MODEL=bisenetv2,KIND=boundary,SCALE=0.25 scripts/cluster/m11_train.sh
#
# MODEL: pidnet_s | bisenetv2.  KIND: uniform | multi | boundary | error.  SCALE: 0.25 0.5 0.75 1.
# At SCALE 1 a warp run magnifies (D_MAX=2 is set automatically unless D_MAX is given).
# At SCALE >= 0.75 the batch is 4 x 2 accumulation steps (spec), else 8.
# EXTRA: more arguments for `method11 train` (gate 0: EXTRA="--lr 0 --max-steps 20").
# Data: ~/avrh_data/cityscapes_train.zip holding leftImg8bit/train and gtFine/train
# (made on the laptop with scripts/get_val.py --train, then zipped and copied).
# Output: ~/avrh_results/method11/runs/<MODEL>_<KIND>_c<SCALE>[_<TAG>]/ with weights.pt at the end.
# Resumable per epoch: a job that hits the time limit is submitted again with the same variables.
set -euo pipefail
: "${MODEL:?}" "${KIND:?}"
SCALE=${SCALE:-1}
ARGS=(--model "$MODEL" --kind "$KIND" --scale "$SCALE" --workers 6)
if [ "$KIND" = boundary ] || [ "$KIND" = error ]; then
  if [ -z "${D_MAX:-}" ] && [ "$SCALE" = 1 ]; then D_MAX=2; fi
  if [ -n "${D_MAX:-}" ]; then ARGS+=(--d-max "$D_MAX"); fi
fi
if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.75 or sys.argv[2] == 'multi' else 1)" "$SCALE" "$KIND"; then
  ARGS+=(--batch 4 --accum 2)
fi
RUN=$HOME/avrh_results/method11/runs/${MODEL}_${KIND}_c${SCALE}${TAG:+_$TAG}

module purge
module load python/3.12.3
unset PYTHONPATH
source ~/avrh-venv/bin/activate

CS=/tmp/avrh_cs_$SLURM_JOB_ID
trap 'rm -rf "$CS"' EXIT
mkdir -p "$CS"
python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
    "$HOME/avrh_data/cityscapes_train.zip" "$CS"
export PIDNET_S_WEIGHTS=$HOME/avrh_data/PIDNet_S_Cityscapes_val.pt PYTHONUNBUFFERED=1

cd ~/avrh
git log --oneline -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -m pytest -q tests/test_method11.py

mkdir -p "$RUN"
python -u -m scripts.method11 train "${ARGS[@]}" --cityscapes "$CS" --run-dir "$RUN" ${EXTRA:-}
echo JOB_OK
