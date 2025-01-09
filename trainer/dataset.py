import transformers
from transformers.trainer_pt_utils import LabelSmoother
import datasets
import torch
from torch.utils.data import Dataset, DataLoader
from trainer import QwenEncoder, LlamaEncoder


class CustomDataset(Dataset):
    """
    For cookbook adapter
    """
    def __init__(self, data, tokenizer, select_len):
        self.tokenizer = tokenizer
        self.data = data
        self.default_instruction = '''You are an AI assitant, you are required to help people solve their problems.'''

        self.select_len = select_len
        if "qwen".upper() in tokenizer.name_or_path.upper():
            self.encoder = QwenEncoder(tokenizer, select_len)
        elif "llama".upper() in tokenizer.name_or_path.upper():
            self.encoder = LlamaEncoder(tokenizer, select_len)
        else:
            raise "This Model is not supported yet"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        reflection = f"\n[Hint]: {item['reflection']}"
        question = f"\n[Question]: {item['question']}"
        instruction = item['instruction'] if 'instruction' in item else self.default_instruction
        answer = item["answer"]

        item = self.encoder.encode_train(instruction=instruction, question=question, reflection=reflection, answer=answer)

        return item

    def collate_fn(self, batch):
        def statisc_ids(input_ids, masks, labels):
            max_length = max([len(i) for i in input_ids])
            return_attention_masks = []
            return_ids = []
            return_masks = []
            return_labels = []

            for ids, mask, label in zip(input_ids, masks, labels):
                padding_num = max_length - len(ids)
                return_ids.append(ids + [self.tokenizer.pad_token_id] * padding_num)
                return_masks.append(mask + [0] * padding_num)
                return_attention_masks.append([1] * len(ids) + [0] * padding_num)
                return_labels.append(label + [LabelSmoother.ignore_index] * padding_num)

            return return_ids, return_attention_masks, return_masks, return_labels

        input_ids = []
        masks = []
        labels = []

        for ones in batch:
            one = ones[0]
            input_ids += [one['input_ids'], one['reflection_input_ids']]
            masks += [one['input_mask'], one['reflection_mask']]
            labels += [one['labels'], one['reflection_labels']]

        input_ids, attention_mask, masks, labels = statisc_ids(input_ids, masks, labels)

        input_ids = torch.tensor(input_ids)
        full_attention_mask = torch.tensor(attention_mask)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        mask = torch.tensor(masks)

        # mask the reflective units of previous layer
        attention_mask[0::2, ...][mask[0::2, ...] == 2] = 0

        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "mask": mask, "full_attention_mask": full_attention_mask}

