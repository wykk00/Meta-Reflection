import re
import math
from typing import Union

def get_feedback(predict, answer) -> Union[bool, str]:
    # successfully solve
    if is_correct(predict, answer):
        return True, f"Student answer: {predict}\n"
    
    feedback_str = f"Student answer: {predict}\nIt's incorrect.\n"

    return False, feedback_str

def extract_answer(s):

    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
    else:
        last_digit = None
        print(f"No digits found in {s!r}", flush=True)
        
    return last_digit

def is_correct(completion, answer):
    gold = extract_answer(answer)
    assert gold is not None, "No ground truth answer found in the document."

    def number_equal(answer, pred):
        if pred is None:
            return False
        try:
            return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)
        except:
            print(
                f"cannot compare two numbers: answer={answer}, pred={pred}", flush=True
            )
            return False

    return number_equal(gold, extract_answer(completion))