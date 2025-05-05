# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import fitz
import torch
import argparse

def loadPDFtoTXT(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Accept a string input.")
    parser.add_argument("--pathToPdf", type=str, help="Path to pdf")
    parser.add_argument("--pathToModel", type=str, help="Path to model checkpoint or huggingface model name")
    args = parser.parse_args()   
    pathToPDF = args.pathToPDF
    logging.basicConfig(
        filename='runPaper.log',      # Log file name
        level=logging.INFO,              # Log level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("starting")


    model_name =  args.pathToModel #"/srv/beegfs02/scratch/mgroepl_master_data/data/Qwen/Qwen/Qwen2.5-3BInstructFullPaperBiggerLr/checkpoint-8000" 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    correct = 0

    instruction_following = "You are Qwen, created by Alibaba Cloud. You are the chair of a very prestige scientific conference in the field of Machine Learning. " \
    "The user send a Scientific Paper to you and you should evaluate it and either accept or reject it. You should do this by giving your opinion on scientific novelty and impactfullness, correctness and how well written it is inside </think>thought process here<think>. " \
    "This should then be directly followed by a clear statement if you accept the paper or reject it inside </answer>accept/reject<anwer>. For example: Assistant: </think>I like this paper. novelty: this work is original enough to be accepted. " \
    "correctness: It sound correct. writing: It is not so well written.<think></answer>Accept<answer>"

    question = loadPDFtoTXT(pathToPDF) +  " Now please give your thoughs and evaluation with the previously mentioned procedure. "\
    "For example:  </think>I like this paper. novelty: this work is original enough to be accepted. " \
    "correctness: It sound correct. writing: It is not so well written.<think></answer>Accept<answer>"

    systemPrompt = "" 
    userPrompt = ""
    messages = [
        {"role": "system", "content": instruction_following},
        {"role": "user", "content": question}
        ]
    with torch.no_grad():
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1000
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


        logging.info("RESPONSE: \n " +response)




    
