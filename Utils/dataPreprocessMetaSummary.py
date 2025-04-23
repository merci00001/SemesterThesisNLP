##This creates a parquet file for training with GRPO. It uses a free Deepseek V3 api to create summaries out of the papers

import re
import os
import datasets

from sklearn.model_selection import train_test_split
import argparse
import requests
import pandas as pd

def summarize_research_paper(text):
  
    OPENROUTER_API_KEY = "APIKEYHERE"

    # OpenRouter API URL
    API_URL = "https://llm.chutes.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-ai/DeepSeek-V3-0324",  # Use OpenAI's GPT-4 via OpenRouter  featherless/qwerky-72b:free  deepseek/deepseek-chat-v3-0324:free
        "messages": [
            {"role": "system", "content": "The user sends you a scientific paper. Your job is to summarize the results, experiments done and everything else important. The summary doesnt need to be short. Also please only include the summary in your response and nothing else like \"here is your summary!\" "},
            {"role": "user", "content": text}
        ],
        "temperature": 0.8,  # Adjust randomness (0 = strict, 1 = creative)
        "max_tokens": 1500  # Limit output length
    }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            return response.json()["choices"][0]["message"]["content"]
        except KeyError:
            print(response)
            return None
    else:
        return f"Error {response.status_code}: {response.text}"


def process_fn(example):


    instruction_following = "<|im_start|>system\n You are Qwen, created by Alibaba Cloud. You are the chair of a very prestige scientific conference in the field of Machine Learning. " \
    "The user send a Scientific Paper to you and you should evaluate it and either accept or reject it. You should do this by giving your opinion on scientific novelty and impactfullness, correctness and how well written it is inside </think>thought process here<think>. " \
    "This should then be directly followed by a clear statement if you accept the paper or reject it inside </answer>accept/reject<anwer>. For example: Assistant: </think>I like this paper. novelty: this work is original enough to be accepted. " \
    "correctness: It sound correct. writing: It is not so well written.<think></answer>Accept<answer><|im_end|>\n"
    
    
    summarizedPaper = summarize_research_paper(example["Paper"])
    if summarizedPaper == None:
        return None
    question = instruction_following + '<|im_start|>user\n' + summarizedPaper +  " Now please give your thoughs and evaluation with the previously mentioned procedure. "\
    "For example:  </think>I like this paper. novelty: this work is original enough to be accepted. " \
    "correctness: It sound correct. writing: It is not so well written.<think></answer>Accept<answer><|im_end|>\n <|im_start|>Assistant\n"
    data = {
        "prompt":  question,
        "ground_truth":
            str(example["decision"]) 

        
    }
    return data



if __name__ == '__main__':
    data_sources = ['ICLR2020.parquet']
    df = pd.concat([pd.read_parquet("/scratch/mgroepl/PaperData/" + file) for file in data_sources], ignore_index=True)
    dfFinal = pd.DataFrame()
    for index, row in df.iterrows():
        print(index)
        data = process_fn(row)
        if data:
            dfFinal = pd.concat([dfFinal,pd.json_normalize(data)],ignore_index = True)
    print(dfFinal.head())
    print(df.shape)
    local_dir="/itet-stor/mgroepl/net_scratch/PaperData/"
    ##dfFinal.to_parquet(os.path.join("/scratch/mgroepl/PaperData/", 'dataset.parquet'))
    train_dataset, test_dataset = train_test_split(dfFinal, test_size=0.05, random_state=42)
    train_dataset.to_parquet(os.path.join(local_dir, 'eval2020Summary.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test2020Summary.parquet'))
