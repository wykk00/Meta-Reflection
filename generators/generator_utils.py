from generators.instruction import *
from generators.model import Message
from typing import Union, List, Optional, Callable, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from .model import generate_chat
import re

def generate_action(
    question: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    strategy: str,
    task: str,
    self_reflection=None,
    max_tokens: int = 2048,
    temperature: float = 0,
    top_p: float = 0,
    top_k: int = 1,
    do_sample: bool = False
) -> Union[str, List[str]]:
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and self_reflection is None:
        raise ValueError(
            f"Invalid arguments: given `strategy=reflexion` but `self_reflection` is None")

    if task == 'math':
        INSTRUCTION = MATH_SIMPLE_ACTION_INSTRUCTION
    elif task == 'programming':
        INSTRUCTION = PY_SIMPLE_ACTION_INSTRUCTION
    elif task == 'ecid':
        INSTRUCTION = ECID_SIMPLE_ACTION_INSTRUCTION


    if strategy == "reflexion":
        system_prompt = INSTRUCTION
        question_input = f"Here are the question and correponding reflection from past trials:\n[Question]: {question}\[Reflection]: {self_reflection}"
    else:
        system_prompt = INSTRUCTION
        question_input = f"[Question]: {question}"
        
    messages = [
            Message(
                role="system",
                content=system_prompt,
            ),
            Message(
                role="user",
                content=question_input
            )
        ]

    response = generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample
    )
    
    return response


def generate_self_reflection(
        question: str,
        feedbacks: Tuple[bool, str],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_tokens: int = 2048,
        temperature: float = 0,
        top_p: float = 0,
        top_k: int = 1,
        do_sample: bool = False
) -> str:
    if task == 'math':
        if feedbacks[0]:
            system_prompt = MATH_SELF_REFLECTION_INSTRUCTION_CORRECT
        else:
            system_prompt = MATH_SELF_REFLECTION_INSTRUCTION
    elif task == 'programming':
        if feedbacks[0]:
            system_prompt = PY_SELF_REFLECTION_INSTRUCTION_CORRECT
        else:
            system_prompt = PY_SELF_REFLECTION_INSTRUCTION
    elif task == 'ecid':
        if feedbacks[0]:
            system_prompt = ECID_SELF_REFLECTION_INSTRUCTION_CORRECT
        else:
            system_prompt = ECID_SELF_REFLECTION_INSTRUCTION
    
    
    feedback = feedbacks[1]

    reflection_input = f"[Question]: {question}\n[Trials and Feedback]: {feedback}\n"

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content=reflection_input
        )
    ]

    response = generate_chat(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample
    )

    return response  



def parse_code_block(string: str) -> Optional[str]:
    """
    Extract code body for programming task.
    """
    code_pattern = r"```python\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)

    if match:
        return match.group(1).strip()

    generic_code_pattern = r"```\n(.*?)\n```"
    match = re.search(generic_code_pattern, string, re.DOTALL)

    if match:
        return match.group(1).strip()

    # return original string
    return string.strip()