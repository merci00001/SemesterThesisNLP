##Converts the downloaded papers into the correct format for finetuning. Make sure the papers have at each at least one review

import re
import os
import datasets

from sklearn.model_selection import train_test_split
import argparse

import pandas as pd

def process_fn(example):

    x = example["0"]
    review = x["review"]
    decision = x["rating"]
    instruction_following = "<|im_start|>system\n You are Qwen, created by Alibaba Cloud. You are the chair of a very prestige scientific conference in the field of Machine Learning. " \
    "The user send a Scientific Paper to you and you should evaluate it and either accept or reject it. You should do this by giving your opinion on scientific novelty and impactfullness, correctness and how well written it is inside </think>thought process here<think>. " \
    "This should then be directly followed by a clear statement if you accept the paper or reject it inside </answer>accept/reject<anwer>. For example: Assistant: </think>I like this paper. novelty: this work is original enough to be accepted. " \
    "correctness: It sound correct. writing: It is not so well written.<think></answer>Accept<answer><|im_end|>\n"

    question = instruction_following + '<|im_start|>user\n' + example["Paper"][0:50] +  " Now please give your thoughs and evaluation with the previously mentioned procedure. "\
    "For example:  </think>I like this paper. novelty: this work is original enough to be accepted. " \
    "correctness: It sound correct. writing: It is not so well written.<think></answer>Accept<answer><|im_end|>\n <|im_start|>Assistant\n</think>"+review+"<think></answer>"+decision+"<answer><|im_end|><|endoftext|>"
    solution = "</think>" +  "</think>"  + "<answer>" + str(example["mean"]) + "<answer>"
    data = {
        "text":  question,
        "ground_truth":
            str(example["decision"]) 

        
    }
    return data

if __name__ == '__main__':

    ##Set these
    datasource = "/source/to/data"
    data_sources = ['ICLR2019.parquet'] 
    df = pd.concat([pd.read_parquet(datasource + file) for file in data_sources], ignore_index=True)
    dfFinal = pd.DataFrame()
    for index, row in df.iterrows():
        data = process_fn(row)
        dfFinal = pd.concat([dfFinal,pd.json_normalize(data)],ignore_index = True)
    print(dfFinal.head())
    print(df.shape)
    local_dir="/itet-stor/mgroepl/net_scratch/PaperData/"
    ##dfFinal.to_parquet(os.path.join("/scratch/mgroepl/PaperData/", 'dataset.parquet'))
    train_dataset, test_dataset = train_test_split(dfFinal, test_size=0.05, random_state=42)
    train_dataset.to_parquet(os.path.join(local_dir, 'trainFT.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'testFT.parquet'))
