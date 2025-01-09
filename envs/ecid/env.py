"""
Feedback generator
"""
import re
import math
from typing import Union

def get_feedback(predict, answer) -> Union[bool, str]:
    
    # successfully solve
    if is_correct(predict, answer):
        return True, f"匹配过程：{predict}\n"
    
    feedback_str = f"匹配过程：{predict}\n匹配出现了错误。\n"

    return False, feedback_str

def extract_answer(s):
    letters = [c for c in s if c.isupper() or c.islower()]
    
    return letters[-1].upper() if letters else None
        

def is_correct(completion, answer):
    gold = extract_answer(answer)
    assert gold is not None, "No ground truth answer found in the document."

    return gold == extract_answer(completion)