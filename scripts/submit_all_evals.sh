#!/usr/bin/env bash
# Submit one independent Slurm job per experiment for evaluation.
# Usage:  bash scripts/submit_all_evals.sh

set -euo pipefail

cd /fs/ess/PAS2136/fangxun/CSE5439/CSE5439-MedSAM2

CKPT_ROOT="/fs/scratch/PAS3272/liu12122/MedImgSeg/outputs"

EXPERIMENTS=(
    baseline_2GPU
    baseline_2worker_2GPU
    baseline_4GPU
    grad_ckpt_2GPU
    grad_ckpt_4GPU
    backbone_8chunk_2GPU
    backbone_8chunk_4GPU
    no_amp_2GPU
    FSDP_2GPU
    offload_cpu_2GPU
    baseline_sam2large_4GPU
    FSDP_sam2large_4GPU
)

mkdir -p slurm_output

submitted=0
skipped=0

for name in "${EXPERIMENTS[@]}"; do
    ckpt="${CKPT_ROOT}/${name}/checkpoints/checkpoint.pt"
    if [[ ! -f "${ckpt}" ]]; then
        echo "SKIP  ${name}  (checkpoint not found)"
        skipped=$((skipped + 1))
        continue
    fi

    job_id=$(EVAL_NAME="${name}" sbatch \
        --job-name="eval_${name}" \
        --output="slurm_output/eval/medsam2_eval_${name}_%j.out" \
        scripts/sbatch_medsam2_eval.sh \
        | awk '{print $NF}')

    echo "SUBMIT  ${name}  -> job ${job_id}"
    submitted=$((submitted + 1))
done

echo ""
echo "Submitted ${submitted} jobs, skipped ${skipped}."
