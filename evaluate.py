import json
import re
import argparse
import numpy as np
import tqdm
import torch
import transformers
from transformers.generation import GenerationConfig
from models import LlamaForCausalLM, Qwen2ForCausalLM
from models import CodebookAdapterModel
from trainer import LlamaEncoder, QwenEncoder
from generators import MATH_SIMPLE_ACTION_INSTRUCTION, PY_SIMPLE_ACTION_INSTRUCTION, ECID_SIMPLE_ACTION_INSTRUCTION
from envs import programming_is_correct, ecid_is_correct, math_is_correct
import os
import warnings

# Ignore warning report
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

ENCODER_MAP = {
    'qwen2': QwenEncoder,
    'llama3': LlamaEncoder
}

INSTRUCTION = {
    'math': MATH_SIMPLE_ACTION_INSTRUCTION,
    'programming': PY_SIMPLE_ACTION_INSTRUCTION,
    'ecid': ECID_SIMPLE_ACTION_INSTRUCTION
}

def generate_sample(model, tokenizer, inputs):
    cur_len = len(inputs[0]['input_ids'])
    
    inputs = {k: torch.LongTensor(v).unsqueeze(0).to(model.device) for k, v in inputs[0].items()}
    outputs = model.generate(
        **inputs,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
        max_new_tokens=768,
        do_sample=False
    )

    return tokenizer.decode(outputs[0].tolist()[cur_len:], skip_special_tokens=True)

def evaluate(completion, doc, task):
    if task == 'math':
        return math_is_correct(completion, doc['answer'])
    elif task == 'programming':
        from generators.generator_utils import parse_code_block
        func = parse_code_block(completion)
        return programming_is_correct(func, doc["test_code"])
    elif task == 'ecid':
        return ecid_is_correct(completion, doc['answer'])
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Trained codebook checkpoint path",
    )
    parser.add_argument(
        "-f", "--sample-input-file", type=str, help="Input file, `.jsonl` format"
    )
    parser.add_argument(
        "-o", "--sample-output-file", type=str, help="Output file, `.jsonl` format"
    )
    parser.add_argument(
        "-dtype", "--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="torch dtype"
    )
    parser.add_argument(
        "-model", "--model", type=str, choices=["llama3", "qwen2"], help="Base LLMs"
    )
    parser.add_argument(
        "-t", "--task", type=str, choices=["math", "programming", "ecid"], help="Task type"
    )

    args = parser.parse_args()

    print(args)

    model_name = MODEL_MAP[args.model]
    torch_dtype = TORCH_DTYPE[args.dtype]
    model_type = MODEL_TYPE[args.model]
    encoder_type = ENCODER_MAP[args.model]
    instruction = INSTRUCTION[args.task]

    if args.sample_input_file is not None:
        dataset = []
        with open(args.sample_input_file, 'r') as f_c:
            for line in f_c:
                dataset.append(json.loads(line))


    print("Loading tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, bf16=True, use_flash_attn=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model ...")
    
    model = model_type.from_pretrained(
        model_name, device_map='auto', trust_remote_code=True, torch_dtype=torch_dtype
    )
    model = CodebookAdapterModel.from_pretrained(
        model, args.checkpoint_path
    ).eval()


    if args.sample_output_file:
        if os.path.exists(args.sample_output_file):
            os.remove(args.sample_output_file)
        f_output = open(args.sample_output_file, "w", encoding="utf-8")

    acc_res = []
    correct = 0
    progress_bar = tqdm.tqdm(enumerate(dataset), desc="Evaluating...", ncols=100)

    encoder = encoder_type(tokenizer, model.peft_config.select_len)

    for idx, doc in progress_bar:
        success, trials = False, 0

        while not success and trials < 5:
            try:
                question = f"\n[Question]: {doc['question']}"
                if args.task == "programming":
                    question += f"Your code should pass these tests: {doc['test_code']}"

                inputs = encoder.encode_inference(instruction, question)

                completion = generate_sample(model, tokenizer, inputs)
                acc = evaluate(completion=completion, doc=doc, task=args.task)

                doc["completion"] = completion
                doc["acc"] = acc
                success = True

            except Exception as e:
                trials += 1
                print(f"Error : {e}")
                pass
        
        if not success:
            acc = False
            doc["acc"] = False
        
        correct += doc["acc"]
        progress_bar.set_description(f"Success: {correct}, Total: {idx + 1}, Rate: {100 * (correct) / (idx + 1):.2f}%")

        if args.sample_output_file:
            f_output.write(json.dumps(doc, ensure_ascii=False) + "\n")
            f_output.flush()
            
        acc_res.append(acc)


    if args.sample_output_file:
        f_output.close()
        
    print("Result:\t", np.mean(acc_res))
