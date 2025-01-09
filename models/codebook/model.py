# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List

import torch
import torch.nn as nn

import peft
from peft.utils import _get_submodules


from .config import CodebookConfig
from .layer import CodebookAttention
from .utils import is_codebook_trainable

import os

class CodebookAdapterModel(nn.Module):

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.peft_config: CodebookConfig = config
        self._cached_adapters: Dict[str, List] = {}
        self.forward = self.model.forward
        self.add_adapter(config)
        self._mark_only_adaption_prompts_as_trainable(self.model)

    def save_pretrained(self, save_directory):

        peft_config = self.peft_config

        # save only the trainable weights
        state_dict = self.state_dict()
        output_state_dict = {k: state_dict[k] for k in state_dict.keys() if "codebook" in k}

        os.makedirs(save_directory, exist_ok=True)

        pth_file = os.path.join(save_directory, 'adapter_model.pth')

        torch.save(output_state_dict, pth_file)

        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True

        if peft_config.task_type is None:
            # deal with auto mapping
            base_model_class = self.model.__class__
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }

        peft_config.save_pretrained(save_directory, auto_mapping_dict=auto_mapping_dict)

        peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(cls, model_id, model_path, is_trainable=False):
        config = CodebookConfig.from_pretrained(model_path)

        # create model
        model = cls(model_id, config)

        adapters_weights = torch.load(os.path.join(model_path, 'adapter_model.pth'), map_location='cpu')

        model.load_state_dict(adapters_weights, strict=False)

        if not is_trainable:
            for _, p in model.named_parameters():
                if p.requires_grad:
                    p.requires_grad = False

        return model

    def generate(
            self, 
            input_ids, 
            attention_mask, 
            eos_token_id, 
            max_new_tokens=512, 
            **kwargs
        ):
        generated = input_ids
        attention_mask = attention_mask
        full_attention_mask = kwargs['full_attention_mask']
        input_mask = kwargs['input_mask']
        past_key_values = None

        for _ in range(max_new_tokens): 
            torch.cuda.empty_cache()

            if past_key_values is not None:
                input_ids = generated[:, -1].unsqueeze(-1)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                input_mask=input_mask,
                full_attention_mask=full_attention_mask,
                past_key_values=past_key_values,
            )
            next_token_logits = outputs.logits[:, -1, :]

            past_key_values = outputs['past_key_values']

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.LongTensor([[1]]).to(attention_mask.device)], dim=-1)
            full_attention_mask = torch.cat([full_attention_mask, torch.LongTensor([[1]]).to(attention_mask.device)], dim=-1)
            input_mask = torch.cat([input_mask, torch.LongTensor([[0]]).to(attention_mask.device)], dim=-1)

            if next_token in eos_token_id:
                break

        return generated

    
    def add_adapter(self, config: CodebookConfig) -> None:

        parents = []
        for name, _ in self.model.named_modules():
            if name.endswith('self_attn'):
                par, _, _ = _get_submodules(self.model, name)
                parents.append(par)
        if len(parents) < config.inserted_layer:
            raise ValueError(
                f"Config specifies more adapter layers '{config.inserted_layer}'"
                f" than the model has '{len(parents)}'."
            )

        parents = parents[config.inserted_layer - 1]

        self._create_adapted_attentions(config, parents)

    def _create_adapted_attentions(self, config: CodebookConfig, parents: nn.Module) -> None:
        """Wrap Attention modules with newly created CodebookAttention modules."""
        attn = CodebookAttention(
            select_len=config.select_len,
            codebook_size=config.codebook_size,
            model=getattr(parents, 'self_attn'),
        )
        setattr(parents, 'self_attn', attn)


    def _mark_only_adaption_prompts_as_trainable(self, model: nn.Module) -> None:
        """Freeze all parameters of the model except the Codebook."""
        for n, p in model.named_parameters():
            if not is_codebook_trainable(n):
                p.requires_grad = False

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            # This is necessary as e.g. causal models have various methods that we
            # don't want to re-implement here.
            if name == "model":  
                raise
            return getattr(self.model, name)
