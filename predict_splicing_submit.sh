#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --partition=gpu-single 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,gpumem_per_gpu:40GB
#SBATCH --mem=40gb
#SBATCH --time=08:00:00
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

# Work directory
WORK_DIR=${HOME}/projects/alphagenome_pytorch/

# Inputs
SUBSET="full"
SPECIES="mouse_human"
KB=""
if [ "$KB" != "" ]; then
    SUBSET="${SUBSET}_${KB}kb"
fi
TRAINING_CONFIG=${WORK_DIR}/configs/splice_finetune_${SUBSET}.yaml
MODEL_DIR=${HOME}/sds/sd17d003/Anamaria/alphagenome_pytorch/${SUBSET}/${SPECIES}/
CHECKPOINT=${MODEL_DIR}/finetune_heads.pt
OUTPUT_DIR=${MODEL_DIR}/predictions_finetune_heads
mkdir -p ${OUTPUT_DIR}
echo "Starting prediction job at "$(date)
echo "Training config: ${TRAINING_CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Model directory: ${OUTPUT_DIR}"

# Train the model
if [ "$KB" != "" ]; then
    PREDICT_SCRIPT=${WORK_DIR}/predict_splicing_windows.py
else
    PREDICT_SCRIPT=${WORK_DIR}/predict_splicing_gene.py
fi
python -u ${PREDICT_SCRIPT} \
  --config ${TRAINING_CONFIG} \
  --checkpoint ${CHECKPOINT}\
  --output-dir ${OUTPUT_DIR} \
  --batch_size 16 \
  > ${OUTPUT_DIR}/predict_${TIMESTAMP}.log
echo "Prediction completed at "$(date)