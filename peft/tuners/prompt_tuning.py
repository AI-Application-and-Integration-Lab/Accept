# coding=utf-8
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

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn as nn

from ..utils import PeftType, PromptLearningConfig


class PromptTuningInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


@dataclass
class PromptTuningConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEmbedding`].

    Args:
        prompt_tuning_init (Union[[`PromptTuningInit`], `str`]): The initialization of the prompt embedding.
        prompt_tuning_init_text (`str`, *optional*):
            The text to initialize the prompt embedding. Only used if `prompt_tuning_init` is `TEXT`.
        tokenizer_name_or_path (`str`, *optional*):
            The name or path of the tokenizer. Only used if `prompt_tuning_init` is `TEXT`.
    """

    prompt_tuning_init: Union[PromptTuningInit, str] = field(
        default=PromptTuningInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    prompt_tuning_init_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    pretrain_init: Union[PromptTuningInit, str] = field(
        default=False, 
        metadata={"help": "How to initialize the PQPA tuning parameters"},
    )
    pretrain_scpp_ckpt: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained prepended prompt."}
    )
    pretrain_scap_ckpt: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained added prompt."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    sub_dim_scpp: int = field(default=32, metadata={"help": "The length of the codewords of prompts."})
    codebook_size_scpp: int = field(default=20, metadata={"help": "Number of codewords in each codebook of prompts."})
    scpp: bool = field(
        default=True,
        metadata={"help": "Whether to apply product quantization to prompts"},
    )
    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_TUNING


class PromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import PromptEmbedding, PromptTuningConfig

    >>> config = PromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     prompt_tuning_init="TEXT",
    ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config, word_embeddings):
        super().__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.num_codebooks_prompt = int(config.token_dim / config.sub_dim_scpp)
        self.scpp = config.scpp
        self.codebook_size_scpp = config.codebook_size_scpp
        self.sub_dim_scpp = config.sub_dim_scpp

        # added prompt in front of the input text
        if not config.scpp:    
            self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim, device="cuda")
        else:
            # (60, 24, 20)
            self.prompt_weight = nn.Parameter(torch.zeros(
                total_virtual_tokens, self.num_codebooks_prompt, config.codebook_size_scpp).cuda())
            nn.init.kaiming_uniform_(self.prompt_weight.data, a=math.sqrt(5))

            # (24, 20, 32)
            self.codebook_prompt = nn.Parameter(torch.zeros(
                self.num_codebooks_prompt, config.codebook_size_scpp, config.sub_dim_scpp).cuda())
            nn.init.kaiming_uniform_(self.codebook_prompt.data, a=math.sqrt(5))

        if config.prompt_tuning_init == PromptTuningInit.TEXT:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]

            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)
        if config.pretrain_init:
            import os
            from ..utils import WEIGHTS_NAME

            pretrain_prompt_filename = os.path.join(config.pretrain_scpp_ckpt, WEIGHTS_NAME)
            prompt_state_dict = torch.load(pretrain_prompt_filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.prompt_weight.data = prompt_state_dict['prompt_weight']
            self.codebook_prompt.data = prompt_state_dict['codebook_prompt']

    def forward(self, indices):
        # Just get embeddings
        if self.scpp:
            # (40, 48, 8)
            prompt_weight = self.prompt_weight.unsqueeze(-1)
            # (40, 48, 16)
            prompt_quantized = torch.sum(prompt_weight * torch.unsqueeze(self.codebook_prompt, 0), dim=2)
            # (40, 768)
            self.embedding = prompt_quantized.view(self.total_virtual_tokens, -1)

            prompt_embeddings = self.embedding.unsqueeze(0).repeat(indices.shape[0], 1, 1)
        else:
            # (32, 40, 768)
            prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
