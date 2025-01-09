import jsonlines
import json
from strategy import run_reflexion
import transformers
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_iters", type=int, help="The maximum number of self-improvement iterations", default=4
    )
    parser.add_argument(
        "--actor_checkpoint", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Checkpoint for Actor LLMs"
    )
    parser.add_argument(
        "--reflector_checkpoint", type=str, default="Qwen/Qwen2-72B-Instruct", help="Checkpoint for Reflector LLMs"
    )
    parser.add_argument(
        "--task", type=str, choices=["math", "programming", "ecid"], help="Task type"
    )
    parser.add_argument(
        "--sample-input-file", type=str, help="Input file, `.jsonl` format"
    )
    parser.add_argument(
        "--sample-output-file", type=str, help="Output file, `.jsonl` format"
    )
    args = parser.parse_args()
    return args

def load_dataset(file_name):
    dataset = []
    with open(file_name, 'r') as f_c:
        for line in f_c:
            item = json.loads(line)
            dataset.append(item)
    return dataset

def load_model(model_checkpoint):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_checkpoint, 
        device_map='auto', 
        torch_dtype="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)

    return model, tokenizer


def main(args):
    # Loading data
    dataset = load_dataset(args.sample_input_file)
    print("## Loading dataset successfully")

    actor_model, actor_tokenizer = load_model(args.actor_checkpoint)
    reflector_model, reflector_tokenizer = load_model(args.reflector_checkpoint)
    print("## Loading models successfully")

    run_reflexion(
        dataset=dataset,
        max_iters=args.max_iters,
        output_file=args.sample_output_file,
        actor_model=actor_model,
        actor_tokenizer=actor_tokenizer,
        reflector_model=reflector_model,
        reflector_tokenizer=reflector_tokenizer,
        task=args.task                    
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
