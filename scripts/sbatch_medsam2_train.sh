#!/usr/bin/env bash
#SBATCH --account=pas3272
#SBATCH --partition=quad
#SBATCH --job-name=medsam2_train
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liu.12122@osu.edu
#SBATCH --output=slurm_output/medsam2_train_%j.out

set -euo pipefail
set -x

cd /fs/ess/PAS2136/fangxun/CSE5439/CSE5439-MedSAM2

NUM_GPUS="${SLURM_GPUS_ON_NODE:-2}"
NUM_NODES="${SLURM_NNODES:-1}"

# Distributed init for multi-GPU on one node (train.py --use-cluster 0)
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

# If Slurm did not set GPU visibility, pin 0..N-1 (matches single_node_train_medsam2.sh style)
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

echo "Master node: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Number of nodes: ${NUM_NODES}"
echo "GPUs per node: ${NUM_GPUS}"

# Default training config and log dir
CONFIG="configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml"
OUTPUT_PATH="/fs/scratch/PAS3272/liu12122/MedImgSeg/outputs/sam2.1_hiera_tiny512_FLARE_RECIST"

# Weights & Biases: set project (required to enable W&B). Name is optional.
# Authenticate with `export WANDB_API_KEY=...` or `wandb login` before submitting.
WANDB_PROJECT="MedSAM2"
WANDB_NAME="baseline_2GPU"
WANDB_ENTITY=""

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT}" ]]; then
  WANDB_ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_NAME}" ]]; then
  WANDB_ARGS+=(--wandb-name "${WANDB_NAME}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi

python training/train.py \
  -c "${CONFIG}" \
  --output-path "${OUTPUT_PATH}" \
  --use-cluster 0 \
  --num-gpus "${NUM_GPUS}" \
  --num-nodes "${NUM_NODES}" \
  --master-addr "${MASTER_ADDR}" \
  --main-port "${MASTER_PORT}" \
  "${WANDB_ARGS[@]}"

echo "training done"
