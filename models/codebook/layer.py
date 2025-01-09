import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import hyperfanin_init_weight, RMSNorm


class CodebookAttention(nn.Module):
    """This module wraps a Attention module and injects Codebook."""

    def __init__(self, model, codebook_size: int, select_len: int):
        """
        Initialize object.

        Args:
            codebook_size: The size of the codebook.
            select_len: The length of the selected units.
            model: The original transformer attention module that is being wrapped.
        """
        assert not isinstance(model, CodebookAttention)
        super().__init__()
        self.model = model
        self.codebook_size = codebook_size
        self.select_len = select_len
        # Assume all parameters of the attention model we are wrapping are on the same device.
        device = next(model.parameters()).device

        target_dtype = (
                model.q_proj.weight.dtype if model.q_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        )

            
        self.codebook_prompt = nn.Parameter(
            torch.empty(1, codebook_size, self.model.hidden_size, device=device, dtype=torch.float32).normal_()
        )

        MLP_HIDDEN_SIZE = 128
        DROPOUT_RATE = 0.1

        layers_encoder = [
            torch.nn.Linear(self.model.hidden_size, MLP_HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=DROPOUT_RATE),
            torch.nn.Linear(MLP_HIDDEN_SIZE, self.model.hidden_size),
        ]

        layers_decoder = [
            torch.nn.Linear(self.model.hidden_size, MLP_HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=DROPOUT_RATE),
            torch.nn.Linear(MLP_HIDDEN_SIZE, self.model.hidden_size),
        ]

        self.attn_dropout = torch.nn.Dropout(p=DROPOUT_RATE)

        self.codebook_encoder = torch.nn.Sequential(*layers_encoder).to(device)
        self.codebook_decoder = torch.nn.Sequential(*layers_decoder).to(device)
        self.codebook_norm = RMSNorm(self.model.hidden_size).to(device)
        

    def forward(self, **kwargs):
        """
        Forward pass for Cookbook
        """

        hidden_states = kwargs['hidden_states']
        input_mask = kwargs['input_mask']
        full_attention_mask = kwargs['full_attention_mask']

        bsz, seq_len, _ = hidden_states.size()

        # When inference, compute only once
        if seq_len == input_mask.size(1):

            question_mask = input_mask == 1
            reflection_mask = input_mask == 2

            # mean Pooling for question, [bs, d]
            question_embedding = self._mean_pooling(hidden_states, question_mask)
            question_embedding = self.codebook_encoder(question_embedding)

            # calculating simlarity score [1, bs, k]
            previous_dtype = question_embedding.dtype

            codebook_prompt = self.codebook_prompt.to(previous_dtype)
            codebook_prompt = self.codebook_decoder(codebook_prompt)

            similarity_score = torch.matmul(question_embedding, codebook_prompt.transpose(-2, -1).to(previous_dtype)) / math.sqrt(self.codebook_size)
            similarity_score = self.attn_dropout(similarity_score)
            similarity_score = F.log_softmax(similarity_score, dim=-1, dtype=torch.float32)

            for bs in range(bsz):
                # sample top-k and replace original tokens
                index = self._gumbel_softmax_topk(
                    logits=similarity_score[0, bs, :],
                    k=self.select_len
                )
                hidden_states[bs][reflection_mask[bs]] = self.codebook_norm(codebook_prompt[index == 1]).to(hidden_states.dtype)

        
        # change attention mask and hidden states
        kwargs['hidden_states'] = hidden_states
        kwargs['attention_mask'] = full_attention_mask

        output, _, past_key_value = self.model(**kwargs)

        return output, None, past_key_value


    def _mean_pooling(self, model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging   
        token_embeddings = model_output # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
            torch.clamp(input_mask_expanded.sum(1), min=torch.finfo(input_mask_expanded.dtype).smallest_normal)


    def _gumbel_softmax_topk(self, logits: torch.Tensor, k: int, tau: float = 1, dim: int = -1, hard : bool = True):
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.topk(k)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft

        ret = ret.unsqueeze(0)
        return ret
