#!/bin/bash

#SBATCH --output=/home/mgroepl/log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/mgroepl/log/%j.out   # where to store error messages
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:a100_80gb:1   #a100_80gb a6000
# Load Conda (Important for Non-Interactive Shells)

#set your conda source here:
source /itet-stor/mgroepl/net_scratch/conda/etc/profile.d/conda.sh
conda init bash
conda activate hFace

#Optional. Set this to avoid saving hugginface models onto your home drive
export HF_HOME=/itet-stor/mgroepl/net_scratch/hCache

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun accelerate launch \
    --mixed_precision bf16 \ 
    --num_processes 1 \
    /home/mgroepl/RESTORE/hFace/test2.py



echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"



# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
