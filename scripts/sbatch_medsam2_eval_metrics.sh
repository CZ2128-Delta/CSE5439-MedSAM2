#!/usr/bin/env bash
#SBATCH --account=pas3272
#SBATCH --partition=quad
#SBATCH --job-name=medsam2_metrics
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liu.12122@osu.edu
#SBATCH --output=slurm_output/medsam2_metrics_%j.out

set -euo pipefail
set -x

cd /fs/ess/PAS2136/fangxun/CSE5439/CSE5439-MedSAM2

NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"

CKPT_ROOT="/fs/scratch/PAS3272/liu12122/MedImgSeg/outputs"
VAL_NPZ="/fs/scratch/PAS3272/liu12122/MedImgSeg/FLARE-Task1-PancancerRECIST-to-3D/validation_npz"

declare -A EXPERIMENTS
EXPERIMENTS["baseline_2GPU"]="configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml|${CKPT_ROOT}/baseline_2GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["baseline_2worker_2GPU"]="configs/exp/2workers/sam2.1_hiera_tiny512_FLARE_RECIST.yaml|${CKPT_ROOT}/baseline_2worker_2GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["baseline_4GPU"]="configs/exp/4GPUs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml|${CKPT_ROOT}/baseline_4GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["grad_ckpt_2GPU"]="configs/exp/sam2.1_hiera_tiny512_FLARE_RECIST_grad_ckpt.yaml|${CKPT_ROOT}/grad_ckpt_2GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["grad_ckpt_4GPU"]="configs/exp/4GPUs/sam2.1_hiera_tiny512_FLARE_RECIST_grad_ckpt.yaml|${CKPT_ROOT}/grad_ckpt_4GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["backbone_8chunk_2GPU"]="configs/exp/sam2.1_hiera_tiny512_FLARE_RECIST_backbone_8chunk.yaml|${CKPT_ROOT}/backbone_8chunk_2GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["backbone_8chunk_4GPU"]="configs/exp/4GPUs/sam2.1_hiera_tiny512_FLARE_RECIST_backbone_8chunk.yaml|${CKPT_ROOT}/backbone_8chunk_4GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["no_amp_2GPU"]="configs/exp/no_amp/sam2.1_hiera_tiny512_FLARE_RECIST.yaml|${CKPT_ROOT}/no_amp_2GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["FSDP_2GPU"]="configs/exp/sam2.1_hiera_tiny512_FLARE_RECIST_FSDP.yaml|${CKPT_ROOT}/FSDP_2GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["offload_cpu_2GPU"]="configs/exp/sam2.1_hiera_tiny512_FLARE_RECIST_offload_cpu.yaml|${CKPT_ROOT}/offload_cpu_2GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["baseline_sam2large_4GPU"]="configs/exp/sam2_large/sam2.1_hiera_large512_FLARE_RECIST.yaml|${CKPT_ROOT}/baseline_sam2large_4GPU/checkpoints/checkpoint.pt"
EXPERIMENTS["FSDP_sam2large_4GPU"]="configs/exp/sam2_large/sam2.1_hiera_large512_FLARE_RECIST_FSDP.yaml|${CKPT_ROOT}/FSDP_sam2large_4GPU/checkpoints/checkpoint.pt"

EVAL_NAME="${EVAL_NAME:-}"

run_metrics() {
    local name="$1"
    local config="$2"
    local ckpt="$3"
    local output="${CKPT_ROOT}/${name}/eval_metrics"

    if [[ ! -f "${ckpt}" ]]; then
        echo "SKIP ${name}: checkpoint not found at ${ckpt}"
        return
    fi

    echo "====== Computing metrics: ${name} ======"

    python training/eval_metrics.py \
        -c "${config}" \
        --checkpoint "${ckpt}" \
        --val-npz-folder "${VAL_NPZ}" \
        --output-path "${output}" \
        --num-gpus "${NUM_GPUS}"

    echo "====== Done: ${name} ======"
}

if [[ -n "${EVAL_NAME}" ]]; then
    IFS='|' read -r config ckpt <<< "${EXPERIMENTS[${EVAL_NAME}]}"
    run_metrics "${EVAL_NAME}" "${config}" "${ckpt}"
else
    for name in "${!EXPERIMENTS[@]}"; do
        IFS='|' read -r config ckpt <<< "${EXPERIMENTS[${name}]}"
        run_metrics "${name}" "${config}" "${ckpt}"
    done
fi

echo "all metric evaluations done"
