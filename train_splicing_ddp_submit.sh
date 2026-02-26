#!/bin/bash
#SBATCH --job-name=train-ddp
#SBATCH --partition=gpu-single 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:4
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.log
#SBATCH --error=slurm_%j.err
# 
# Helix GPU options:
# - A40 (48 GB):   --gres=gpu:A40:1
# - A100 (40 GB):  --gres=gpu:A100:1
# - A100 (80 GB):  --gres=gpu:A100:1
# - H200 (141 GB): --gres=gpu:H200:1

set -e

# Initialize conda for bash shell
source ${HOME}/miniforge3/etc/profile.d/conda.sh

# Load CUDA module before activating conda environment
module load devel/cuda
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Fix PyTorch memory fragmentation (reduces reserved-but-unallocated memory)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
conda activate alphagenome_pytorch

# Verify CUDA setup
echo "CUDA setup verification:"
echo "  CUDA_HOME: ${CUDA_HOME}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# Exit if CUDA is not available
python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "ERROR: CUDA is not available in PyTorch!"
    exit 1
}

# Create a timestamp for unique log file names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Splicevo directory
WORK_DIR=${HOME}/projects/alphagenome_pytorch/

# Inputs
SUBSET="adult"
SPECIES="mouse_human"
KB="50"
TRAINING_CONFIG=${WORK_DIR}/configs/splice_finetune_${SUBSET}_${KB}kb.yaml
MODEL_DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_pytorch/${SUBSET}_${KB}kb/${SPECIES}_ddp/
mkdir -p ${MODEL_DIR}
echo "Starting training job at "$(date)
echo "Training config: ${TRAINING_CONFIG}"
echo "Model directory: ${MODEL_DIR}"

# Train the model with mmultiple GPUs
torchrun --nproc_per_node=4 ${WORK_DIR}/train_splicing_ddp.py \
        --config ${TRAINING_CONFIG} \
        --ddp -u \
  > ${MODEL_DIR}/train_${TIMESTAMP}.log
echo "Training completed at "$(date)