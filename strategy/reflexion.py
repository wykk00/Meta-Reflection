from generators.generator_utils import generate_action, generate_self_reflection
from generators.instruction import *
from typing import List
import os
import json
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from envs import math_get_feedback, ecid_get_feedback, programming_get_feedback

def run_reflexion(
    dataset: List[dict],
    max_iters: int,
    output_file: str,
    actor_model: PreTrainedModel,
    actor_tokenizer: PreTrainedTokenizer,
    reflector_model: PreTrainedModel,
    reflector_tokenizer: PreTrainedTokenizer,
    task: str
) -> None:
    f_o = open(output_file, 'w')

    for idx, item in tqdm(enumerate(dataset)):
        is_solved = False
        cur_iter = 1
        reflections = []
        responses = []
        feedbacks = []
        try:
            question = item["question"]

            if task == "programming":
                question += f"Your code should pass these tests: {item['test_code']}"

            # first attempt
            cur_response = generate_action(
                question=question,
                model=actor_model,
                tokenizer=actor_tokenizer,
                strategy="simple",
                task=task
            )

            responses.append(cur_response)
            
            judge, feedback = get_feedback(
                response=cur_response,
                doc=item,
                task=task
            )
            is_solved = False
            feedbacks.append(feedback)
            
            cur_feedbacks = (judge, feedback)

            # use self-reflection to iteratively improve
            while not is_solved and cur_iter <= max_iters:
                # get reflection
                reflection = generate_self_reflection(
                    question=question,
                    feedbacks=cur_feedbacks,
                    model=reflector_model,
                    tokenizer=reflector_tokenizer,
                    task=task
                )

                reflections += [reflection]

                # apply self-reflection in the next attempt
                cur_response = generate_action(
                    question=question,
                    self_reflection=reflection,
                    model=actor_model,
                    tokenizer=actor_tokenizer,
                    strategy="reflexion",
                    task=task
                )
                responses.append(cur_response)

                judge, feedback = get_feedback(
                    response=cur_response,
                    doc=item,
                    task=task
                )
                is_solved = judge
                feedbacks.append(feedback)
                cur_feedbacks = (judge, feedback)

                cur_iter += 1

        except Exception as e:
            print(f"Case {idx}:  {e}")
            continue
        
        # store successful QRA (Question-Reflection-Answer) triplets that resulted in correct answers
        if is_solved:
            if task == 'math':
                instruction = MATH_SIMPLE_ACTION_INSTRUCTION
            elif task == 'programming':
                instruction = PY_SIMPLE_ACTION_INSTRUCTION
            elif task == 'ecid':
                instruction = ECID_SIMPLE_ACTION_INSTRUCTION

            save_dict = {
                'reflection': reflections[-1],
                'question': question,
                'answer': responses[-1],
                'instruction': instruction
            }

            f_o.write(json.dumps(save_dict, ensure_ascii=False) + "\n")
            f_o.flush()

    f_o.close()


def get_feedback(response, doc, task):
    if task == 'math':
        return math_get_feedback(response, doc['answer'])
    elif task == 'programming':
        from generators.generator_utils import parse_code_block
        func = parse_code_block(response)
        return programming_get_feedback(func, doc["test_list"])
    elif task == 'ecid':
        return ecid_get_feedback(response, doc['answer'])
    else:
        raise NotImplementedError()