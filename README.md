# Paper prediction Semester Thesis Repo

This repository contains all code to train and run the models for paper acceptance prediction. It also includes instructions for setting everything up and the api code to fetch papers with all the relevant information from OpenReview.

# Setup

## Bash scripts
There are a number of bash scripts to run the code on the slurm cluster inside the /Scripts folder.
The a100_80gb.sh runs the script on the 80gb GPU and the a6000.sh on the 50gb one. Inside the script, you have to set your conda source. Additionally, you can set where huggingface saves their models to avoid saving everything to your home directory.
The python file runs via accelerate srun accelerate launch \   here come accelerate args like deepspeed config and mixed precision \ yourScript.py --yourscriptArgsHere.
Accelerate is a library that automatically enables multi gpu when using huggingfaces TRL library. If you wish to run multi gpu, you can use multiGPU.sh. You just have to set num_processes to the amount of GPU specified (https://huggingface.co/docs/trl/main/en/deepspeed_integration). 



## Enviroment installation

First install conda. To install the correct enviroment on slurm, run the installEnv.sh script as a job. You have to set the correct paths inside the batchfile (where to store the packages and your path to your conda installation).
This will install TRL and deepspeed. This should be enough to run the model, but chances are, some packages have to be added. 

##Models

The model trained through biased papers but still performed the best can be found here: https://polybox.ethz.ch/index.php/s/9zGSRwbia5HcmM2

## Known issues

There existed a bug when trying to finetune a model with TRL (maybe they fixed it), where the .map function didnt correctly map a tensor. There exists a fix on github
