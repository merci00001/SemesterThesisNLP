#This scripts runs the model on the ARC-Challenge dataset. The prompt can be changed on line 25-33. I used to run it on SLURM and used the SLURM .out file as a log.
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge") #Maxwell-Jia/AIME_2024   openai/gsm8k

train_dataset = dataset["train"]
eval_dataset = dataset["test"]


##Set which model to test here
model_name = "Model name or path here"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

correct = 0
for x in eval_dataset:
    systemPrompt=     "You are Qwen, created by Alibaba Cloud. You are a student trying to solve a logical problem. " \
        "The user sends such a problem to you and you should evaluate it and give the solution to the problem. You should do this by giving your thought process how to solve it inside </think>thought process here<think>. " \
        "This should then be directly followed by the solution to the problem </answer>solution<anwer>"
    alph = [" A: "," B: "," C: "," D: "," E: "," F: "," G: "]
    problem = x["question"]
    choices = ""
    for y in range(len(x["choices"]["text"])):
        choices += alph[y] + ": " + x["choices"]["text"][y] 
    question = systemPrompt  + problem + ". Please choose the answer from: "+ choices + "and answer with A, B,C or D depending on which answer you choose."+ " Now please give your thoughs and evaluation as detailed and structured as possible. "
    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": question}
        ]

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

    print("QUESTION: \n " +problem + choices)
    print("RESPONSE: \n " +response)
    print("SOLUTION: \n " +x["answerKey"])

    
