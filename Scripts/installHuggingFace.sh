#!/bin/bash
#SBATCH --output=/home/mgroepl/log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/mgroepl/log/%j.out   # where to store error messages
# Load Conda (adjust path if needed)
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere

echo "Running on node: $(hostname)"
sinfo --Format nodehost:20,features_act:80 |grep -v '(null)' |awk 'NR == 1; NR > 1 {print $0 | "sort -n"}'
nvcc -V
export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf
nvidia-smi

source /itet-stor/mgroepl/net_scratch/conda/etc/profile.d/conda.sh
conda init bash
export CONDA_PKGS_DIRS=/itet-stor/mgroepl/net_scratch/cache
TMPDIR=/itet-stor/mgroepl/net_scratch/cache


# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Create Conda environment (if it doesn't exist)
ENV_NAME="hFace"

# Activate the environment

CONDA_OVERRIDE_CUDA=11.8 conda create --name $ENV_NAME pytorch torchvision pytorch-cuda=11.8 --channel pytorch --channel nvidia
conda activate $ENV_NAME
pip3 install trl
# pip3 install flash-attn --no-build-isolation
echo done
exit 0
