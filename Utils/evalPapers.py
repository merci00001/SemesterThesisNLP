from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import torch

def getScore(input,gT):
    scores = []
    reject = None
    for x in input:
        res = x[-80:].split("</answer>")

        if len(res)==2:
            answer = res[1]
            reject = re.search(r"\b([1-5])\b", answer)
            accept = re.search(r"\b(10|[6-9])\b", answer)
            gAccept = re.search(r"accept", gT,  re.IGNORECASE)
            gReject = re.search(r"reject", gT,  re.IGNORECASE)
            if accept and gAccept:
                scores.append(1)
            elif reject and gReject:
                scores.append(-1)
            else:
                scores.append(0)
    return [scores, reject]
logging.basicConfig(
    filename='PaperEval3BRLFullPapernumGen1AggrCORRECTModel.log',      # Log file name
    level=logging.INFO,              # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("starting")

data_files = {"train": "train2023.parquet"}
dataset = load_dataset("parquet", data_files="/itet-stor/mgroepl/net_scratch/PaperData/train2023.parquet")['train'] ##train2023  eval2023Summary


num_generations = 1
model_name = "/srv/beegfs02/scratch/mgroepl_master_data/data/Qwen/Qwen/Qwen2.5-3BInstructSummary/checkpoint-6000"  #/srv/beegfs02/scratch/mgroepl_master_data/data/Qwen/Qwen/Qwen2.5-3BInstructSummary/checkpoint-6000, /srv/beegfs02/scratch/mgroepl_master_data/data/Qwen/Qwen/Qwen2.5-3BInstructFullPaperBiggerLrCORRECT/checkpoint-8500
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

correct = 0
total = 0
confusion = {"TP": 0,
             "FP" : 0,
             "TN": 0,
             "FN":0}
logging.info("starting")
for x in dataset:
    try:
        y = x["ground_truth"]
        total +=1
        responses = []
        print("starting")
        model_inputs = tokenizer([x["prompt"]], return_tensors="pt").to(model.device)
        u = 0
        failed = 0
        scores = []
        while u < num_generations:
            with torch.no_grad():
                logging.info("generate")

                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=2000,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                logging.info(response)
                t = getScore([response],y)
                score = t[0]
                reject = t[1]
                if len(score) == 1:
                    scores.append(score[0])
                    u+=1
                else:
                    failed +=1
                if failed >=5:
                    break
        if failed >=5:
            continue
        logging.info(scores)

        value = sum(scores)/len(scores)
        if value > 0.5:
            confusion["TP"] += 1
            
        elif value < -0.5:
            correct +=1
            confusion["TN"] +=1

        elif value < 0.5 and value>=0 and reject:
            confusion["FN"] +=1
        else:
            confusion["FP"] +=1
        logging.info(confusion)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"[Warning] Skipping batch  due to CUDA OOM.")
            torch.cuda.empty_cache()  # optional: clean up memory
            continue
        else:
            raise 
logging.info("end")
