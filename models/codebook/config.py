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

from dataclasses import dataclass, field

from peft.config import PeftConfig
from peft.utils import PeftType

import os
import json

@dataclass
class CodebookConfig(PeftConfig):

    codebook_size: int = field(default=None, metadata={"help": "Size of codebook"})
    inserted_layer: int = field(default=None, metadata={"help": "Position of inserted layer"})
    select_len: int = field(default=None, metadata={"help" : "Length of soft prompt to insert"})

    def __post_init__(self):
        self.peft_type = PeftType.ADAPTION_PROMPT

    @classmethod
    def from_pretrained(cls, path_name: str):
        config_file = os.path.join(path_name, 'adapter_config.json')

        with open(config_file, 'r') as f_c:
            data = json.load(f_c)

        config = cls(
            codebook_size=data['codebook_size'],
            inserted_layer=data['inserted_layer'],
            select_len=data['select_len'],
        )

        return config

    @property
    def is_adaption_prompt(self) -> bool:
        """Return True if this is an adaption prompt config."""
        return True