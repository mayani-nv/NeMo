# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from transformers import AutoModelForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

@dataclass
class Phi3Config(GPTConfig):
    normalization: str = "LayerNorm"
    activation_func: Callable = F.gelu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False

@dataclass 
class Phi3ConfigSmall(Phi3Config):
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int =  14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    rotary_base:float = 1000000
    vocab_size: int = 100352
    layernorm_epsilon: float = 1.0e-05
    initializer_range: float =  0.02
    max_position_embeddings: int = 8192
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    bf16: bool = True
    params_dtype=torch.bfloat16
    

class Phi3Model(GPTModel):
    def __init__(
        self,
        config: Optional[Phi3Config] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        super().__init__(config or Phi3Config(), optim=optim, tokenizer=tokenizer, model_transform=model_transform)

@io.model_importer(Phi3Model, "hf")
class HFPhi3Importer(io.ModelConnector["AutoModelForCausalLM", Phi3Model]):
    def init(self) -> Phi3Model:
        from transformers import AutoTokenizer
        return Phi3Model(self.config, tokenizer=self.tokenizer)
    

    def apply(self, output_path: Path) -> Path:
        from transformers import AutoModelForCausalLM

        # Check if the source is valid model identifier or path
        try:
            source = AutoModelForCausalLM.from_pretrained(str(self), torch_dtype='auto', trust_remote_code=True)
        except Exception as e:
            raise ValueError(f"Failed to load the model from source '{self}': {e}")
        
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Phi3 model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path
    
    def convert_state(self, source, target):
        mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.query_key_value.weight": "decoder.layers.*.self_attention.linear_qkv.weight",
            "model.layers.*.self_attn.dense.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.up_proj.weight": "decoder.layers.*.mlp.linear_fc1.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.final_layernorm.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",

            # bias related mapping
            "model.layers.*.self_attn.query_key_value.bias": "decoder.layers.*.self_attention.linear_qkv.bias",
            "model.layers.*.self_attn.dense.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            # "model.layers.*.self_attn.rotary_emb.inv_freq":  #this can be ignored
            "model.layers.*.mlp.up_proj.bias":"decoder.layers.*.mlp.linear_fc1.bias",
            "model.layers.*.mlp.down_proj.bias":"decoder.layers.*.mlp.linear_fc2.bias",
            "model.layers.*.input_layernorm.bias": "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias" ,
            "model.layers.*.post_attention_layernorm.bias": "decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "model.final_layernorm.bias": "decoder.final_layernorm.bias",
        }
        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv, _import_qkv_bias, _import_linear_fc1])
    
    @property
    def tokenizer(self):
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)), trust_remote_code=True)

        
    
    @property
    def config(self) -> Phi3Config:
        # output = Phi3ConfigSmall()
        from transformers import AutoConfig
       
        source = AutoConfig.from_pretrained(str(self), trust_remote_code=True)
    
        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base
        output = Phi3ConfigSmall(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.layer_norm_epsilon,
            rotary_base=source.rope_embedding_base,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            add_bias_linear = True,
            add_qkv_bias= True
        )
        print ("output: ", output)
        return output
    
@io.state_transform(
# source_key="model.layers.*.self_attn.qkv_proj.weight", # this is for phi-3-mini-4k instruct
source_key="model.layers.*.self_attn.query_key_value.weight",
target_key="decoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv(ctx: io.TransformCTX, qkv_weight):
    megatron_config = ctx.target.config

    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num //  num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    old_tensor_shape = qkv_weight.size()
    new_q_tensor_shape = (head_num, head_size, old_tensor_shape[1])
    new_kv_tensor_shape = (num_query_groups, head_size, old_tensor_shape[1])
    q, k, v = qkv_weight.split(
        [head_num * head_size, num_query_groups * head_size, num_query_groups * head_size], dim=0
    )
    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights = torch.empty((0, head_size, old_tensor_shape[1])).type_as(qkv_weight)
    for i in range(num_query_groups):
        qkv_weights = torch.cat((qkv_weights, q[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
        qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
        qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights

@io.state_transform(
   # source_key="model.layers.*.self_attn.qkv_proj.weight",
   source_key="model.layers.*.self_attn.query_key_value.bias",
   target_key="decoder.layers.*.self_attention.linear_qkv.bias", 
)
def _import_qkv_bias(ctx: io.TransformCTX, hf_qkv_bias):
    megatron_config = ctx.target.config
    head_num = megatron_config.num_attention_heads
    num_query_groups = megatron_config.num_query_groups
    heads_per_group = head_num //  num_query_groups
    hidden_size = megatron_config.hidden_size
    head_size = megatron_config.kv_channels

    new_q_tensor_shape = (head_num, head_size)
    new_kv_tensor_shape = (num_query_groups, head_size)
    q, k, v = hf_qkv_bias.split(
        [head_num * head_size, num_query_groups * head_size, num_query_groups * head_size], dim=0
    )
    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_bias = torch.empty((0, head_size)).type_as(hf_qkv_bias)
    for i in range(num_query_groups):
        qkv_bias = torch.cat((qkv_bias, q[i * heads_per_group : (i + 1) * heads_per_group, :]))
        qkv_bias = torch.cat((qkv_bias, k[i : i + 1, :]))
        qkv_bias = torch.cat((qkv_bias, v[i : i + 1, :]))

    qkv_bias = qkv_bias.reshape(
        [
            head_size * (head_num + 2 * num_query_groups),
        ]
    )
    return qkv_bias


@io.state_transform(
    source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"), # phi-3-mini-4k-instruct
    # source_key = ("model.layers.*.mlp.up_proj.weight", "model.layers.*.mlp.down_proj.weight"),  # phi-3-small-8k-instruct
    target_key="decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)

__all__ = [
    "Phi3Config",
    "Phi3ConfigSmall",
    "Phi3Model"
]
