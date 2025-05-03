# Paper prediction Semester Thesis Repo

This repository contains all code to train and run the models for paper acceptance prediction. It also includes instructions for setting everything up and the api code to fetch papers with all the relevant information from OpenReview.

# Setup

## Bash scripts
There are a number of bash scripts to run the code on the slurm cluster inside the /Scripts folder.
The a100_80gb.sh runs the script on the 80gb GPU and the a6000.sh on the 50gb one. Inside the script, you have to set your conda source. Additionally, you can set where huggingface saves their models to avoid saving everything to your home directory.
The python file runs via accelerate srun accelerate launch \   <<<here come script args and accelerate args like deepspeed config and mixed precision>>> \ yourScript.py
Accelerate is a library that automatically enables multi gpu when using huggingfaces TRL library. If you wish to run multi gpu, you can use multiGPU.sh. You just have to set num_processes to the amount of GPU specified. 



##Enviroment installation

