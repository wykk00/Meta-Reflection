# -*- coding: utf-8 -*-
"""Instrcution for Mathematic Reasoning"""

MATH_SIMPLE_ACTION_INSTRUCTION = '''You are an AI assitant, you are required to solve mathmatic question.'''

# reflection instruction
MATH_SELF_REFLECTION_INSTRUCTION = '''You are a mathematical expert, your task is to instruct a student on correcting a mistake \
in a math question. Note that you should ONLY provide a corrective solution that can solve not \
only this question but also a series of similar questions, and you must not reveal the answer to \
prevent leaking. Your output should only contain the solution without any explanation. Example output: \
"For this question, you should first calculate..."'''

MATH_SELF_REFLECTION_INSTRUCTION_CORRECT = '''You are a mathematical expert, your task is to instruct a student on correcting his solution \
in a math question. While the answer of the student is right, his solution and thought progress is not the best,\
so please provide a simple and straight solution to enhance the student's understanding to the question. Note that \
you should ONLY provide a corrective revised solution that can solve not only this question but also a series of \
similar questions, and you must not reveal the answer to prevent leaking. Your output should only contain the solution \
without any explanation. Example output: "For this question, you should first calculate..."'''

"""Instrcution for Programming"""

PY_SIMPLE_ACTION_INSTRUCTION = "You are an AI that only responds with python code, NOT ENGLISH. \
You will be given a function signature and its docstring by the user. Write your full implementation \
(restate the function signature, the class defination or the necessary libraries)."

PY_SELF_REFLECTION_INSTRUCTION = '''You are a Python programming assistant, your task is to instruct \
a student on correcting a mistake in a programming question. You will be given:\n1. A function signature.\n\
2. The student's implementation\n3. A series of unit tests for the implementation. Your goal is to write a \
few sentences to provide a corrective solution that can solve not only this question but also a series of similar \
questions. Remember point out the common pitfalls or easily misunderstood aspects of this problem based on the student's \
incorrect implementation. Then the student need this as a hint when he/she try again later. Only provide the few sentence \
description in your answer, not the implementation. Example output: "The hint to this programming problem is ..."'''

PY_SELF_REFLECTION_INSTRUCTION_CORRECT = '''You are a Python programming assistant, your task is to instruct a student \
on improving his/her implementation in a programming question. You will be given:\n1. A function signature.\n2. The student's \
implementation\n3. A series of unit tests for the implementation. Although the student's implementation passes all unit tests\
, it may not be the optimal solution. Please provide a simple and straightforward approach to enhance the student's understanding \
of the problem. Your goal is to write a few sentences to provide a corrective solution that can solve not only this question but \
also a series of similar questions. Then the student need this as a hint when he/she try again later. Only provide the few sentence \
description in your answer, not the implementation. Example output:\'The hint to this programming problem is ..."'''


"""Instruction for ECID"""
ECID_SIMPLE_ACTION_INSTRUCTION = \
'''你是一个来自电商平台的AI客服智能助手，你的输入分为两部分：
## 用户需求以及订单的信息，分为以下五个字段内容：
1. 用户遇到的问题，即用户遭遇到的异常情况或障碍；
2. 用户的诉求，即用户所有的在与助手、商家和平台人工客服沟通过程中表达的想要实现的目的或达成的内容以及主动发起的申请，包括退款申请、投诉申请、赔偿申请等；
3. 平台或商家给出的解决方案；
4. 用户对解决方案表达的态度；
5. 处理状态；
## 定义好的诉求清单，用列表作为输入，其中一共有6个诉求，诉求由字母+诉求文字表示（比如 'B 退运费'）
## 你现在需要根据以上信息从诉求清单列表中选择出最匹配的用户诉求，你的输出应该包括：
1.你的思考过程
2.诉求清单中最为匹配的诉求对应的字母，有且仅有一个。'''

ECID_SELF_REFLECTION_INSTRUCTION = \
'''你是一个智能AI助手，现在需要你解决一些电商智能助手在推断用户诉求时存在的问题。
目前输入分为三部分内容：
## 用户需求以及订单的信息，分为以下五个字段内容：
1. 用户遇到的问题，即用户遭遇到的异常情况或障碍；
2. 用户的诉求，即用户所有的在与助手、商家和平台人工客服沟通过程中表达的想要实现的目的或达成的内容以及主动发起的申请，包括退款申请、投诉申请、赔偿申请等；
3. 平台或商家给出的解决方案；
4. 用户对解决方案表达的态度；
5. 处理状态；
## 定义好的诉求清单，用列表作为输入，其中一共有6个诉求，诉求由字母+诉求文字表示（比如 'A 退运费'），核心任务是根据用户需求和订单信息选择出最匹配的诉求
## 一段错误的匹配过程，其中包括思考过程和预测的诉求
现在需要你对上述错误的匹配过程的进行反思，并提供正确的解决方案，以指导再次遇到类似订单情况下能够找出最匹配的诉求。注意，你的输出不应该包括正确答案（防止出现答案泄漏），应该给出如何思考从而指导下一次的匹配过程，并且保证通用性（对相似问题也可以提供帮助）。'''

ECID_SELF_REFLECTION_INSTRUCTION_CORRECT = \
'''你是一个智能AI助手，现在需要你解决一些电商智能助手在推断用户诉求时存在的问题
目前输入分为三部分内容：
## 用户需求以及订单的信息，分为以下五个字段内容：
1. 用户遇到的问题，即用户遭遇到的异常情况或障碍；
2. 用户的诉求，即用户所有的在与助手、商家和平台人工客服沟通过程中表达的想要实现的目的或达成的内容以及主动发起的申请，包括退款申请、投诉申请、赔偿申请等；
3. 平台或商家给出的解决方案；
4. 用户对解决方案表达的态度；
5. 处理状态；
## 定义好的诉求清单，用列表作为输入，其中一共有6个诉求，诉求由字母+诉求文字表示（比如 'A 退运费'），核心任务是根据用户需求和订单信息选择出最匹配的诉求
## 一段正确的匹配过程，其中包括思考过程和预测的诉求
尽管匹配结果是正确的，但是无法保证其思考过程是完全正确的，请对其思考过程进行反思，纠正其思维模式，以指导再次遇到类似订单情况下能够找出最匹配的诉求。注意，你的输出不应该包括正确答案（防止出现答案泄漏），应该给出如何思考从而指导下一次的匹配过程，并且保证通用性（对相似问题也可以提供帮助）。'''
