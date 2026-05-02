#!/usr/bin/env bash
#SBATCH --account=pas3272
#SBATCH --partition=quad
#SBATCH --job-name=medsam2_eval
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liu.12122@osu.edu
#SBATCH --output=slurm_output/medsam2_eval_%j.out

set -euo pipefail
set -x

cd /fs/ess/PAS2136/fangxun/CSE5439/CSE5439-MedSAM2

NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
NUM_NODES="${SLURM_NNODES:-1}"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$(python - <<'EOF'
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

echo "Master node: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Number of nodes: ${NUM_NODES}"
echo "GPUs per node: ${NUM_GPUS}"

# ---------------------------------------------------------------------------
# Configure which experiment to evaluate.
# Set CONFIG to the same Hydra config used during training.
# Set CHECKPOINT to the .pt file produced by that training run.
# ---------------------------------------------------------------------------

CKPT_ROOT="/fs/scratch/PAS3272/liu12122/MedImgSeg/outputs"
VAL_NPZ="/fs/scratch/PAS3272/liu12122/MedImgSeg/FLARE-Task1-PancancerRECIST-to-3D/validation_npz"

# --- Experiments to evaluate (config  |  checkpoint  |  eval output) ---
# Uncomment ONE block at a time, or loop over all (see below).

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

# Set EVAL_NAME to run a single experiment, or leave empty to run all.
EVAL_NAME="${EVAL_NAME:-}"

WANDB_PROJECT="${WANDB_PROJECT:-MedSAM2_eval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

run_eval() {
    local name="$1"
    local config="$2"
    local ckpt="$3"
    local output="${CKPT_ROOT}/${name}/eval"

    if [[ ! -f "${ckpt}" ]]; then
        echo "SKIP ${name}: checkpoint not found at ${ckpt}"
        return
    fi

    echo "====== Evaluating: ${name} ======"

    WANDB_ARGS=()
    if [[ -n "${WANDB_PROJECT}" ]]; then
        WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
        WANDB_ARGS+=(--wandb-name "${name}")
    fi
    if [[ -n "${WANDB_ENTITY}" ]]; then
        WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi

    python training/eval.py \
        -c "${config}" \
        --checkpoint "${ckpt}" \
        --val-npz-folder "${VAL_NPZ}" \
        --output-path "${output}" \
        --num-gpus "${NUM_GPUS}" \
        --num-nodes "${NUM_NODES}" \
        --master-addr "${MASTER_ADDR}" \
        --main-port "${MASTER_PORT}" \
        "${WANDB_ARGS[@]}"

    echo "====== Done: ${name} ======"
}

if [[ -n "${EVAL_NAME}" ]]; then
    IFS='|' read -r config ckpt <<< "${EXPERIMENTS[${EVAL_NAME}]}"
    run_eval "${EVAL_NAME}" "${config}" "${ckpt}"
else
    for name in "${!EXPERIMENTS[@]}"; do
        IFS='|' read -r config ckpt <<< "${EXPERIMENTS[${name}]}"
        run_eval "${name}" "${config}" "${ckpt}"
    done
fi

echo "all evaluations done"
