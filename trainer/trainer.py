import transformers
from transformers import PreTrainedModel
from transformers import Trainer, TrainingArguments, TrainerCallback
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Union, Optional, Dict, List
from transformers.trainer_pt_utils import LabelSmoother


class CustomTrainer(Trainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        ref_model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        mode: Optional[str] = "alignment",
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks
        )

        self.ref_model = ref_model
        self.label_smoother = LabelSmoother(epsilon=0.1) # Label Smoother
        self.mode = mode

    def _mean_pooling(self, model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging   
        token_embeddings = model_output # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=torch.finfo(input_mask_expanded.dtype).smallest_normal)

    def _matrix_diag(self, diagonal):
        N = diagonal.shape[-1]
        shape = diagonal.shape[:-1] + (N, N)
        device, dtype = diagonal.device, diagonal.dtype
        result = torch.zeros(shape, dtype=dtype, device=device)
        indices = torch.arange(result.numel(), device=device).reshape(shape)
        indices = indices.diagonal(dim1=-2, dim2=-1)
        result.view(-1)[indices] = diagonal
        return result

    def sim(self, reps1, reps2):
        reps1_unit = F.normalize(reps1, dim=-1)
        reps2_unit = F.normalize(reps2, dim=-1)
        if len(reps1.shape) == 2:
            sim_mat = torch.einsum("ik,jk->ij", [reps1_unit, reps2_unit])
        elif len(reps1.shape) == 3:
            sim_mat = torch.einsum('bik,bjk->bij', [reps1_unit, reps2_unit])
        else:
            print(f"{len(reps1.shape)} dimension tensor is not supported for this function!")
        return sim_mat

    def emd(self, out1, avg_out1, out2, avg_out2, lamb=20, rescale_ratio=None):
        """
        Shape of each tensor
        out1: [b, s1, d]
        avg_out1: [b, 1, d]
        out2: [b, s2, d]
        avg_out2: [b, 1, d]
        """
        assert out1.shape[0] == out2.shape[0] and avg_out1.shape == avg_out2.shape
        
        # upcast to float32
        out1 = out1.float()
        avg_out1 = avg_out1.float()
        out2 = out2.float()
        avg_out2 = avg_out2.float()


        cost_matrix = 1 - self.sim(out1, out2)
        if rescale_ratio is not None:
            cost_matrix = cost_matrix * rescale_ratio

        # Sinkhorn iteration
        iter_times = 10
        with torch.no_grad():
            r = torch.bmm(out1, avg_out2.transpose(1, 2))
            r[r <= 0] = 1e-8
            r = r / r.sum(dim=1, keepdim=True)
            c = torch.bmm(out2, avg_out1.transpose(1, 2))
            c[c <= 0] = 1e-8
            c = c / c.sum(dim=1, keepdim=True)
            P = torch.exp(-1 * lamb * cost_matrix)
            u = (torch.ones_like(c) / c.shape[1])
            for i in range(iter_times):
                v = torch.div(r, torch.bmm(P, u))
                u = torch.div(c, torch.bmm(P.transpose(1, 2), v))
            u = u.squeeze(dim=-1)
            v = v.squeeze(dim=-1)
            transport_matrix = torch.bmm(torch.bmm(self._matrix_diag(v), P), self._matrix_diag(u))
        assert cost_matrix.shape == transport_matrix.shape

        # emd distance
        emd = torch.mul(transport_matrix, cost_matrix).sum(dim=1).sum(dim=1, keepdim=True)
        return emd

    def compute_loss(self, model, inputs, return_outputs=False):

        # extract input_ids
        model_input_ids = inputs["input_ids"][0::2,...]
        ref_model_input_ids = inputs["input_ids"][1::2,...]

        model_attention_mask = inputs["attention_mask"][0::2,...]
        full_attention_mask = inputs["full_attention_mask"][0::2, ...]
        ref_model_attention_mask = inputs["attention_mask"][1::2,...]
        
        input_mask = inputs["mask"][0::2,...]
        reflection_mask = inputs["mask"][1::2,...]
        labels = inputs["labels"][0::2,...]


        model_output = model(
            input_ids=model_input_ids,
            labels=labels,
            attention_mask=model_attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            input_mask=input_mask,
            full_attention_mask=full_attention_mask
        )

        loss = 0

        if self.mode == 'alignment':
            with torch.no_grad():
                ref_model_output = self.ref_model(
                    input_ids=ref_model_input_ids,
                    attention_mask=ref_model_attention_mask,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True
                )

            num_layers = model.config.num_hidden_layers
            
            last_n_layers = num_layers - model.peft_config.inserted_layer + 1
            
            assert last_n_layers > 0

            batch_size, seq_len, dim = model_output.hidden_states[-1].size()

            # [b * last_n_layers, s, n]
            model_hidden_states = torch.cat(model_output.hidden_states[-last_n_layers:],)
            ref_model_hidden_states = torch.cat(ref_model_output.hidden_states[-last_n_layers:],)
            
            # [b * last_n_layers, s]
            ref_reflection_mask = reflection_mask.repeat_interleave(last_n_layers, dim=0)
            
            model_reflection_mask = input_mask
            model_reflection_mask[model_reflection_mask != 2] = 0
            model_reflection_mask[model_reflection_mask == 2] = 1
            model_reflection_mask = input_mask.repeat_interleave(last_n_layers, dim=0)

            # [b * last_n_layers, 1, n]
            model_mean_pooling = self._mean_pooling(model_hidden_states, model_reflection_mask).unsqueeze(1)
            ref_model_mean_pooling = self._mean_pooling(ref_model_hidden_states, ref_reflection_mask).unsqueeze(1)

            model_hidden_states[model_reflection_mask != 1,] = 0
            ref_model_hidden_states[ref_reflection_mask != 1,] = 0

            emd_loss = self.emd(
                model_hidden_states, model_mean_pooling,
                ref_model_hidden_states, ref_model_mean_pooling
            )
            
            loss = emd_loss.mean()

        elif self.mode == 'sft':
            # SFT loss
            sft_loss = model_output['loss']
            loss = sft_loss

        if return_outputs:
            outputs = {model_output["hidden_states"][-1]}
        
        return (loss, outputs) if return_outputs else loss

