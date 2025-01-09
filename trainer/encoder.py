import transformers
from transformers.trainer_pt_utils import LabelSmoother
import torch

class Encoder():
    def __init__(self, tokenizer, select_len):
        self.tokenizer = tokenizer
        self.select_len = select_len

    
    def encode_train(self, instruction, question, reflection, answer):
        """
        Encode for training phase
        """
        pass

    def encode_inference(self, instruction, question):
        """
        Encode for inference phase
        """
        pass


class LlamaEncoder(Encoder):
    def __init__(self, tokenizer, select_len):
        super().__init__(tokenizer, select_len)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Using Llama Encoder")


    def encode_inference(self, instruction, question):
        IGNORE_TOKEN_ID = [LabelSmoother.ignore_index]
        BOS = [self.tokenizer.bos_token_id]
        START_HEADER_ID = self.tokenizer("<|start_header_id|>", add_special_tokens=False).input_ids
        END_HEADER_ID = self.tokenizer("<|end_header_id|>", add_special_tokens=False).input_ids
        EOT_ID = self.tokenizer("<|eot_id|>", add_special_tokens=False).input_ids 
        NL = self.tokenizer("\n", add_special_tokens=False).input_ids
        SYSTEM = START_HEADER_ID + self.tokenizer("system", add_special_tokens=False).input_ids + END_HEADER_ID + NL + NL
        USER = START_HEADER_ID + self.tokenizer("user", add_special_tokens=False).input_ids + END_HEADER_ID + NL + NL
        ASSISTANT = START_HEADER_ID + self.tokenizer("assistant", add_special_tokens=False).input_ids + END_HEADER_ID + NL + NL
        PAD = [self.tokenizer.pad_token_id]

        input_ids = []

        # mask
        input_mask = []
        attention_mask = []
        full_attention_mask = []

        # add instruction
        input_ids += BOS + SYSTEM + self.tokenizer(instruction, add_special_tokens=False).input_ids + EOT_ID     

        input_mask += [0] * len(input_ids)
        attention_mask += [1] * len(input_ids)
        full_attention_mask += [1] * len(input_ids)

        # add question
        question_enc = self.tokenizer(question, add_special_tokens=False).input_ids

        input_ids += USER + question_enc

        input_mask += [0] * len(USER) + [1] * len(question_enc)
        attention_mask += [1] * len(USER + question_enc)
        full_attention_mask += [1] * len(USER + question_enc)

        # add reflection
        input_ids += self.select_len * PAD + EOT_ID

        input_mask += [2] * self.select_len + [0] * len(EOT_ID)
        attention_mask += [0] * self.select_len + [1] * len(EOT_ID) 
        full_attention_mask += [1] * (self.select_len + len(EOT_ID))

        # add ASSISTANT
        input_ids += ASSISTANT

        input_mask += [0] * len(ASSISTANT)
        attention_mask += [1] * len(ASSISTANT)
        full_attention_mask += [1] * len(ASSISTANT)

        assert len(input_ids) == len(input_mask) == len(attention_mask) == len(full_attention_mask)

        return [{
            'input_ids': input_ids,
            'input_mask': input_mask,
            'attention_mask': attention_mask,
            'full_attention_mask': full_attention_mask
        }]


    def encode_train(self, instruction, question, reflection, answer):
        IGNORE_TOKEN_ID = [LabelSmoother.ignore_index]
        BOS = [self.tokenizer.bos_token_id]
        START_HEADER_ID = self.tokenizer("<|start_header_id|>", add_special_tokens=False).input_ids
        END_HEADER_ID = self.tokenizer("<|end_header_id|>", add_special_tokens=False).input_ids
        EOT_ID = self.tokenizer("<|eot_id|>", add_special_tokens=False).input_ids 
        NL = self.tokenizer("\n", add_special_tokens=False).input_ids
        SYSTEM = START_HEADER_ID + self.tokenizer("system", add_special_tokens=False).input_ids + END_HEADER_ID + NL + NL
        USER = START_HEADER_ID + self.tokenizer("user", add_special_tokens=False).input_ids + END_HEADER_ID + NL + NL
        ASSISTANT = START_HEADER_ID + self.tokenizer("assistant", add_special_tokens=False).input_ids + END_HEADER_ID + NL + NL
        PAD = [self.tokenizer.pad_token_id]

        input_ids = []
        reflection_input_ids = []

        labels = []
        reflection_labels = []

        # input & reflection mask
        input_mask = []
        reflection_mask = []

        # add instruction
        input_ids += BOS + SYSTEM + self.tokenizer(instruction, add_special_tokens=False).input_ids + EOT_ID     
        reflection_input_ids += BOS + SYSTEM + self.tokenizer(instruction, add_special_tokens=False).input_ids + EOT_ID

        labels += IGNORE_TOKEN_ID * len(input_ids)
        reflection_labels += IGNORE_TOKEN_ID * len(reflection_input_ids)

        input_mask += [0] * len(input_ids)
        reflection_mask += [0] * len(input_ids)

        # add question
        question_enc = self.tokenizer(question, add_special_tokens=False).input_ids

        input_ids += USER + question_enc
        reflection_input_ids += USER + question_enc

        labels += IGNORE_TOKEN_ID * len(USER + question_enc)
        reflection_labels += IGNORE_TOKEN_ID * len(USER + question_enc)

        input_mask += [0] * len(USER) + [1] * len(question_enc)
        reflection_mask += [0] * len(USER + question_enc)

        # add reflection
        reflection_enc = self.tokenizer(reflection, add_special_tokens=False).input_ids

        input_ids += self.select_len * PAD + EOT_ID
        reflection_input_ids += reflection_enc + EOT_ID

        labels += IGNORE_TOKEN_ID * (self.select_len + len(EOT_ID))
        reflection_labels += IGNORE_TOKEN_ID * len(reflection_enc) + EOT_ID

        # 2 in question mask denote for reflection, 1 in question mask denote for question
        input_mask += [2] * self.select_len + [0] * len(EOT_ID)
        reflection_mask += [1] * len(reflection_enc) + [0] * len(EOT_ID)

        # add answer
        answer_enc = self.tokenizer(answer, add_special_tokens=False).input_ids

        input_ids += ASSISTANT + answer_enc + EOT_ID
        labels += len(ASSISTANT) * IGNORE_TOKEN_ID + answer_enc + EOT_ID

        input_mask += [0] * len(ASSISTANT + answer_enc + EOT_ID)

        # TODO adding answer into reflection 

        assert len(input_ids) == len(labels) == len(input_mask) 
        assert len(reflection_input_ids) == len(reflection_labels) == len(reflection_mask)

        return [{
            'input_ids': input_ids,
            'reflection_input_ids': reflection_input_ids,
            'labels': labels,
            'reflection_labels': reflection_labels,
            'input_mask': input_mask,
            'reflection_mask': reflection_mask
        }]


class QwenEncoder(Encoder):
    def __init__(self, tokenizer, select_len):
        super().__init__(tokenizer, select_len)
        print(f"Using Qwen Encoder")

    def encode_inference(self, instruction, question):
        IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
        im_start = self.tokenizer(IM_START, add_special_tokens=False).input_ids
        im_end = self.tokenizer(IM_END, add_special_tokens=False).input_ids
        nl_tokens = self.tokenizer(NL, add_special_tokens=False).input_ids
        _system = self.tokenizer('system', add_special_tokens=False).input_ids + nl_tokens
        _user = self.tokenizer('user', add_special_tokens=False).input_ids + nl_tokens
        _assistant = self.tokenizer('assistant', add_special_tokens=False).input_ids + nl_tokens
        pad_tokens = [self.tokenizer.pad_token_id]

        input_ids = []

        # mask
        input_mask = []
        attention_mask = []
        full_attention_mask = []

        # add instruction
        input_ids += im_start + _system + self.tokenizer(instruction, add_special_tokens=False).input_ids + im_end + nl_tokens     

        input_mask += [0] * len(input_ids)
        attention_mask += [1] * len(input_ids)
        full_attention_mask += [1] * len(input_ids)

        # add question
        question_enc = self.tokenizer(question, add_special_tokens=False).input_ids

        input_ids += im_start + _user + question_enc

        input_mask += [0] * len(im_start + _user) + [1] * len(question_enc)
        attention_mask += [1] * len(im_start + _user + question_enc)
        full_attention_mask += [1] * len(im_start + _user + question_enc)

        # add reflection
        input_ids += self.select_len * pad_tokens + im_end + nl_tokens

        input_mask += [2] * self.select_len + [0] * len(im_end + nl_tokens)
        attention_mask += [0] * self.select_len + [1] * len(im_end + nl_tokens) 
        full_attention_mask += [1] * (self.select_len + len(im_end + nl_tokens))

        # add ASSISTANT
        input_ids += im_start + _assistant

        input_mask += [0] * len(im_start + _assistant)
        attention_mask += [1] * len(im_start + _assistant)
        full_attention_mask += [1] * len(im_start + _assistant)

        assert len(input_ids) == len(input_mask) == len(attention_mask) == len(full_attention_mask)

        return [{
            'input_ids': input_ids,
            'input_mask': input_mask,
            'attention_mask': attention_mask,
            'full_attention_mask': full_attention_mask
        }]


    def encode_train(self, instruction, question, reflection, answer,):
        IGNORE_TOKEN_ID = [LabelSmoother.ignore_index]
        IM_START, IM_END, NL = "<|im_start|>", "<|im_end|>", "\n"
        im_start = self.tokenizer(IM_START, add_special_tokens=False).input_ids
        im_end = self.tokenizer(IM_END, add_special_tokens=False).input_ids
        nl_tokens = self.tokenizer(NL, add_special_tokens=False).input_ids
        _system = self.tokenizer('system', add_special_tokens=False).input_ids + nl_tokens
        _user = self.tokenizer('user', add_special_tokens=False).input_ids + nl_tokens
        _assistant = self.tokenizer('assistant', add_special_tokens=False).input_ids + nl_tokens
        pad_tokens = [self.tokenizer.pad_token_id]

        input_ids = []
        reflection_input_ids = []

        labels = []
        reflection_labels = []

        # input & reflection mask
        input_mask = []
        reflection_mask = []

        # add instruction
        input_ids += im_start + _system + self.tokenizer(instruction, add_special_tokens=False).input_ids + im_end + nl_tokens   
        reflection_input_ids += im_start + _system + self.tokenizer(instruction, add_special_tokens=False).input_ids + im_end + nl_tokens

        labels += IGNORE_TOKEN_ID * len(input_ids)
        reflection_labels += IGNORE_TOKEN_ID * len(reflection_input_ids)

        input_mask += [0] * len(input_ids)
        reflection_mask += [0] * len(input_ids)

        # add question
        question_enc = self.tokenizer(question, add_special_tokens=False).input_ids

        input_ids += im_start + _user + question_enc
        reflection_input_ids += im_start + _user + question_enc

        labels += IGNORE_TOKEN_ID * len(im_start + _user + question_enc)
        reflection_labels += IGNORE_TOKEN_ID * len(im_start + _user + question_enc)

        input_mask += [0] * len(im_start + _user) + [1] * len(question_enc)
        reflection_mask += [0] * len(im_start + _user + question_enc)

        # add reflection
        reflection_enc = self.tokenizer(reflection, add_special_tokens=False).input_ids

        input_ids += self.select_len * pad_tokens + im_end + nl_tokens
        reflection_input_ids += reflection_enc + im_end + nl_tokens

        labels += IGNORE_TOKEN_ID * (self.select_len + len(im_end + nl_tokens))
        reflection_labels += IGNORE_TOKEN_ID * len(reflection_enc) + im_end + nl_tokens

        # 2 in question mask denote for reflection, 1 in question mask denote for question
        input_mask += [2] * self.select_len + [0] * len(im_end + nl_tokens)
        reflection_mask += [1] * len(reflection_enc) + [0] * len(im_end + nl_tokens)

        # add answer
        answer_enc = self.tokenizer(answer, add_special_tokens=False).input_ids

        input_ids += im_start + _assistant + answer_enc + im_end + nl_tokens
        labels += len(im_start + _assistant) * IGNORE_TOKEN_ID + answer_enc + im_end + nl_tokens

        input_mask += [0] * len(im_start + _assistant + answer_enc + im_end + nl_tokens)

        # TODO adding answer into reflection 

        assert len(input_ids) == len(labels) == len(input_mask) 
        assert len(reflection_input_ids) == len(reflection_labels) == len(reflection_mask)

        return [{
            'input_ids': input_ids,
            'reflection_input_ids': reflection_input_ids,
            'labels': labels,
            'reflection_labels': reflection_labels,
            'input_mask': input_mask,
            'reflection_mask': reflection_mask
        }]


if __name__ == '__main__':

    # tokenizer = transformers.AutoTokenizer.from_pretrained("qwen/qwen2-7B-Instruct")
    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
 
    encoder = LlamaEncoder(tokenizer, 16)
    
    encoder.encode_train(
        instruction="You are an AI assistant",
        question="\n1+1=?",
        reflection="\nFor this question, you need to first.",
        answer="The answer is 2."
    )

