from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer

data_files = {"train": "trainFT.parquet", "test": "testFT.parquet"}
dataset = load_dataset(
    "/itet-stor/mgroepl/net_scratch/PaperData", data_files = data_files
)

modelname = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname)

instruction_template = "system\n"
response_template = "Assistant\n"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

training_args = SFTConfig(
    output_dir="/srv/beegfs02/scratch/mgroepl_master_data/data/Qwen/Qwen/Qwen2.5-1.5BFT",
    bf16 = True,
    per_device_train_batch_size = 1,
    optim ="schedule_free_sgd"
)
trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    args=training_args,
    data_collator=collator,
    
)
trainer.train()
