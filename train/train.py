from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#set path to data and data names and model to train
pathToModel = "model name or path to model"
outputDir = "/path/to/save/model"
path = "/itet-stor/mgroepl/net_scratch/PaperData/"
data_files = {"train": ["eval2019Summary.parquet","eval2020Summary.parquet","eval2021Summary.parquet","eval2022Summary.parquet"], "test": "test.parquet"}

dataset = load_dataset(
    path, data_files = data_files
)

train_dataset = dataset["train"]
#eval_dataset = dataset["test"]
print(len(train_dataset))


def reward_loose(completions, ground_truth, **kwargs):
    reward = []
    for x,y in zip(completions,ground_truth):
        ##check if answer and think tags exist
        debugAnswer = re.findall(r"</Decision>", x, re.DOTALL)
        if len(debugAnswer) != 1:
            reward.append(0)
            continue
        res = re.findall(r"\*\*Decision:(.*?)\*\*$", x, re.DOTALL) ##"<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$",   
        if len(res)==1:
            #check if its a number between the tags
            answer = res[0][1]
            reject = re.search(r"reject", answer,  re.IGNORECASE)
            accept = re.search(r"accept", answer,  re.IGNORECASE)
            gAccept = re.search(r"accept", y,  re.IGNORECASE)
            gReject = re.search(r"reject", y,  re.IGNORECASE)
            totalReward = 0
            if accept and gAccept:
                totalReward +=0.5
            elif reject and gReject:
                totalReward +=0.5
            else:
                totalReward +=0.1
            nov = re.search(r"novelty:", x,  re.IGNORECASE)
            corr = re.search(r"correctness", x,  re.IGNORECASE)
            wri = re.search(r"writing:", x,  re.IGNORECASE)

            if nov:
                totalReward +=0.1    
            if corr:
                totalReward +=0.1    
            if wri:
                totalReward +=0.1    
            reward.append(totalReward)  
        else:
            reward.append(0)        
    return reward






def reward_len(completions, ground_truth, **kwargs):
    reward = []
    for x,y in zip(completions,ground_truth):
        ##check if answer and think tags exist
        debugThink = re.findall(r"<think>", x, re.DOTALL)
        debugAnswer = re.findall(r"<answer>", x, re.DOTALL)
        if len(debugAnswer) != 1 or len(debugThink)!= 1:
            reward.append(0)
            continue
        res = re.findall(r"</think>(.*?)<think>\s*</answer>(.*?)<answer>\s*$", x, re.DOTALL) ##"<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$",   
        if len(res)==1:
            #check if its a number between the tags
            answer = res[0][1]
            reject = re.search(r"reject", answer,  re.IGNORECASE)
            accept = re.search(r"accept", answer,  re.IGNORECASE)
            gAccept = re.search(r"accept", y,  re.IGNORECASE)
            gReject = re.search(r"reject", y,  re.IGNORECASE)
            if accept and gAccept:
                reward.append(1)
            elif reject and gReject:
                reward.append(1)
            else:
                reward.append(0.1)
        else:
            reward.append(0)        
    return reward


def reward_score_based(completions, ground_truth, **kwargs):
    reward = []
    for x,y in zip(completions,ground_truth):
        ##check if answer and think tags exist
        debugThink = re.findall(r"<think>", x, re.DOTALL)
        debugAnswer = re.findall(r"<answer>", x, re.DOTALL)
        if len(debugAnswer) != 1 or len(debugThink)!= 1:
            reward.append(0)
            continue
        res = re.findall(r"</think>(.*?)<think>\s*</answer>(.*?)<answer>\s*$", x, re.DOTALL) ##"<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$",   
        if len(res)==1:
            #check if its a number between the tags
            answer = res[0][1]
            reject = re.search(r"\b([1-5])\b", answer,  re.IGNORECASE)
            accept = re.search(r"\b(10|[6-9])\b", answer,  re.IGNORECASE)
            gAccept = re.search(r"accept", y,  re.IGNORECASE)
            gReject = re.search(r"reject", y,  re.IGNORECASE)
            if accept and gAccept:
                reward.append(1)
            elif reject and gReject:
                reward.append(1)
            else:
                reward.append(0.1)
        else:
            reward.append(0)        
    return reward

model = AutoModelForCausalLM.from_pretrained(pathToModel) 
print("training " + pathToModel)
training_args = GRPOConfig(output_dir=outputDir, logging_steps=20, log_completions =True,  do_train = True,max_completion_length = 1000, num_generations = 4, num_train_epochs = 3,per_gpu_train_batch_size = 4,bf16 = True)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_score_based,
    args=training_args,
    train_dataset=train_dataset
    #optimizers=(optimizer,None)
    
)
trainer.train()
