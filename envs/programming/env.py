import ast
import signal
import astunparse
from typing import List, Tuple

from .executor_utils import execute_code_with_timeout

def get_feedback(func: str, tests: List[str], timeout: int = 3,) -> Tuple[bool, str]:
    # Combine function code and assert statement
    imports = 'from typing import *\nfrom collections import *\nfrom math import *\n'
    func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

    # Run the tests and collect the results
    success_tests = []
    failed_tests = []
    is_passing = True
    num_tests = len(func_test_list)
    for i in range(num_tests):
        try:

            execute_code_with_timeout(func_test_list[i], timeout)

            success_tests += [tests[i]]
        except Exception:
            
            failed_tests += [f"{tests[i]}"]
            is_passing = False

    state = []
    for test in tests:
        if test in success_tests:
            state += [True]
        else:
            state += [False]

    state = tuple(state)

    feedback = f"Implementation: {func}\n" + "Tests passed:"
    for test in success_tests:
        feedback += f"\n{test}"
    feedback += "\n\nTests failed:"
    for test in failed_tests:
        feedback += f"\n{test}"
        
    return is_passing, feedback

def is_correct(func: str, test: str, timeout: int = 3) -> bool:
    """
    Evaluates the implementation on Human-Eval Python.

    probably should be written in a dataset-agnostic way but not now
    """
    imports = "from typing import *\nfrom collections import *\nfrom math import *\n"

    code = f"""{imports}\n{func}\n{test}""" # Test Code, function + test code (assert)
    try:

        execute_code_with_timeout(code, timeout)

        return True
    except Exception as e:
        return False
