# -*- coding: utf-8 -*-

from typing import List, Union, Optional, Literal
import dataclasses
from transformers import PreTrainedModel, PreTrainedTokenizer
import transformers
import torch
import os
import json

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str

 
def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"

def message_to_dict(message: Message) -> dict:
    return {'role': message.role, 'content': message.content}

def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


def generate_chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Message], 
    max_tokens: int = 2048, 
    temperature: float = 0, 
    top_p: float = 0,
    top_k: int = 1,
    do_sample: bool = False
):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    messages = [message_to_dict(msg) for msg in messages]

    if pipeline.tokenizer.pad_token_id is None:
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

    terminators = [
        pipeline.tokenizer.eos_token_id,
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    completion = outputs[0]["generated_text"][-1]['content']

    return completion
