import transformers
from transformers import TrainingArguments
from trainer import CustomDataset
from trainer import CustomTrainer
from trainer import load_codebook_model
from models import LlamaForCausalLM
from models import Qwen2ForCausalLM
from transformers import TrainerCallback
import os
import torch
import json
import argparse

MODEL_MAP = {
    'llama3': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen2': 'Qwen/Qwen2-7B-Instruct',
}

TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float
}

MODEL_TYPE = {
    'llama3': LlamaForCausalLM,
    'qwen2': Qwen2ForCausalLM
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model", type=str, choices=["llama3", "qwen2"], default="llama3", help="Base LLM")
    parser.add_argument("-dtype", "--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="fp16", help="torch dtype, fp32, fp16 or bf16")
    parser.add_argument("-codebook_size", "--codebook_size", default=1024, type=int, help="Size of cookbook")
    parser.add_argument("-select_len", "--select_len", default=32, type=int, help="Length of reflective units")
    parser.add_argument("-inserted_layer", "--inserted_layer", default=25, type=int, help="Position of inserted layers")
    parser.add_argument("-lr1", "--lr1", type=float, default=5e-5, help="Learning rate for stage 1")
    parser.add_argument("-lr2", "--lr2", type=float, default=1e-5, help="Learning rate for stage 2")
    parser.add_argument("-batch_size", "--batch_size", type=int, default=4)
    parser.add_argument("-gradient_accumulation_steps", "--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("-epochs1", "--epochs1", type=int, default=2, help="Epochs for stage 1")
    parser.add_argument("-epochs2", "--epochs2", type=int, default=3, help="Epochs for stage 2")
    parser.add_argument("-source_file", "--source_file", type=str, help="Training dataset file, `.jsonl` format")
    parser.add_argument("-save_path", "--save_path", type=str, help="Save directory")
    args = parser.parse_args()
    return args


class LogToJSONLCallback(TrainerCallback):
    def __init__(self, output_file):
        self.output_file = output_file

    def on_log(self, args, state, control, logs=None, **kwargs):
        with open(self.output_file, "a") as f:
            f.write(json.dumps(logs) + "\n")

if __name__ == '__main__':
    args = get_args()
    print(args)

    model_name = MODEL_MAP[args.model]
    torch_dtype = TORCH_DTYPE[args.dtype]
    model_type = MODEL_TYPE[args.model]

    print("Loading model...")
    model = model_type.from_pretrained(model_name, device_map='auto', torch_dtype=torch_dtype, trust_remote_code=True)
    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading Codebook model...")
    model = load_codebook_model(model, codebook_size=args.codebook_size, inserted_layer=args.inserted_layer, select_len=args.select_len)
    
    print("Loading reflection model...")
    ref_model = model_type.from_pretrained(model_name, device_map='auto', torch_dtype=torch_dtype, trust_remote_code=True)
    ref_model.eval()

    for param in ref_model.parameters():
        param.requires_grad = False

    file_name = args.source_file

    data = []

    with open(file_name, 'r') as f_c:
        for line in f_c:
            data.append(json.loads(line))

    dataset = CustomDataset(data, tokenizer, args.select_len)

    save_path = args.save_path

    if save_path is None:
        save_path = f'saves/{args.model}/layer_{args.inserted_layer}_codebooksize_{args.codebook_size}_selectlen_{args.select_len}/'

    os.makedirs(save_path, exist_ok=True,)

    log_file = os.path.join(save_path, 'logs.jsonl')

    if os.path.exists(log_file):
        os.remove(log_file)

    callback = LogToJSONLCallback(log_file)
    # stage 1, Meta-Reflection Alignment:
    print(f"#### Stage 1 Training ####")

    emd_training_args = TrainingArguments(
            output_dir='./saves', # redirect to null
            overwrite_output_dir=True,
            report_to="none",
            bf16=True if torch_dtype == "bf16" else False,  # Use BF16 if available
            dataloader_pin_memory=False,
            logging_strategy="steps",
            logging_steps=10,
            lr_scheduler_type="linear",
            save_strategy="no",
            optim="adamw_torch_fused",
            warmup_ratio=0.1,
            learning_rate=args.lr1,
            num_train_epochs=args.epochs1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 3,
            gradient_checkpointing=False,
        )

    emd_trainer = CustomTrainer(
        model=model,
        ref_model=ref_model,
        args=emd_training_args,
        data_collator=dataset.collate_fn,
        train_dataset=dataset,
        tokenizer=tokenizer,
        mode='alignment',
        callbacks=[callback]
    )

    emd_trainer.train()
    torch.cuda.empty_cache()
    
    # stage 2, SFT Loss:
    print(f"#### Stage 2 Training ####")

    # Save memory
    del ref_model
    torch.cuda.empty_cache()

    sft_training_args = TrainingArguments(
            output_dir='./saves', # redirect to null
            overwrite_output_dir=True,
            report_to="none",
            bf16=True if torch_dtype == "bf16" else False,  # Use BF16 if available
            dataloader_pin_memory=False,
            logging_strategy="steps",
            logging_steps=10,
            lr_scheduler_type="linear",
            save_strategy="no",
            optim="adamw_torch_fused",
            warmup_ratio=0.1,
            learning_rate=args.lr2,
            num_train_epochs=args.epochs2,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 3,
            gradient_checkpointing=False,
        )

    sft_trainer = CustomTrainer(
        model=model,
        ref_model=None,
        args=sft_training_args,
        data_collator=dataset.collate_fn,
        train_dataset=dataset,
        tokenizer=tokenizer,
        mode='sft',
        callbacks=[callback],
    )

    sft_trainer.train()
    
    # save model
    sft_trainer.model.save_pretrained(save_path)
    sft_trainer.tokenizer.save_pretrained(save_path)

    # save config
    with open(os.path.join(save_path, 'args.json'), 'w') as f_c:
        json.dump(vars(args), f_c, indent=4)
