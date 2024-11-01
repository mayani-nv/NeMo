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
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

@dataclass
class Phi3Config(GPTConfig):
    normalization: str = "RMSNorm"
    activation_func: Callable = F.gelu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    

@dataclass
class Phi3Config4B(Phi3Config):
    num_layers: int = 32
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 32
    num_query_groups: int = 32
    rotary_base: float = 10000.0
    vocab_size: int = 32064

@dataclass
class Phi3Config8B(Phi3Config):
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int =  14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    rotary_base: float = 10000.0
    vocab_size: int = 100352
    layernorm_epsilon: float = 1.0e-05
    initializer_range: float =  0.02
    max_position_embeddings: int = 8192
    add_bias_linear: bool = True


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
        from transformers import Phi3ForCausalLM, AutoModelForCausalLM

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
        # Define mapping for mini-4k-instruct
        # mapping = {
        #     "model.embed_tokens.weight": "embedding.word_embeddings.weight",
        #     "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
        #     "model.layers.*.self_attn.qkv_proj.weight": "decoder.layers.*.self_attention.linear_qkv.weight",
        #     "model.layers.*.mlp.gate_up_proj.weight": "decoder.layers.*.mlp.linear_fc1.weight",
        #     "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
        #     "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
        #     "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
        #     "model.norm.weight": "decoder.final_layernorm.weight",
        #     "lm_head.weight": "output_layer.weight",
        # }
        # Definining mapping for small-8k-instruct model
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
            "model.layers.*.self_attn_query_key_value.bias": "decoder.layers.*.self_attention.linear_qkv.bias",
            "model.layers.*.self_attn.dense.bias": "decoder.layers.*.self_attention.linear_proj.bias",
            # "model.layers.*.self_attn.rotary_emb.inv_freq":  #this can be ignored
            "model.layers.*.mlp.up_proj.bias":"decoder.layers.*.mlp.linear_fc1.bias",
            "model.layers.*.mlp.down_proj.bias":"decoder.layers.*.mlp.linear_fc2.bias",
            "model.layers.*.input_layernorm.bias": "decoder.layers.*.self_attention.linear_qkv.layer_norm_bias" ,
            "model.layers.*post_attention_layernorm.bias": "decoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "model.final_layernorm.bias": "decoder.final_layernorm.bias",
        }

        # Print keys for debugging
        source_keys_layers_0 = [key for key in source.state_dict().keys() if key.startswith("model.layers.0.")]
        target_keys_layers_0 = [key for key in target.state_dict().keys() if key.startswith("module.decoder.layers.0.")]
        print("Source state dict keys:", source_keys_layers_0)
        print("Target state dict keys:", target_keys_layers_0)

        # Print keys not associated with layers.0
        source_keys_non_layers_0 = [key for key in source.state_dict().keys() if not key.startswith("model.layers.")]
        target_keys_non_layers_0 = [key for key in target.state_dict().keys() if not key.startswith("module.decoder.layers.")]
        print("Source state dict keys (non-layers.0):", source_keys_non_layers_0)
        print("Target state dict keys (non-layers.0):", target_keys_non_layers_0)

        # Check dimensions and existence
        for src_key, tgt_key in mapping.items():
            src_key_specific = src_key.replace('*', '0')
            tgt_key_specific = tgt_key.replace('*', '0')

            try:
                if src_key_specific in source.state_dict():
                    src_shape = source.state_dict()[src_key_specific].shape
                    print(f"Source shape for {src_key_specific}: {src_shape}")
                else:
                    print(f"Source key not found: {src_key_specific}")

                if tgt_key_specific in target.state_dict():
                    tgt_shape = target.state_dict()[tgt_key_specific].shape
                    print(f"Target shape for {tgt_key_specific}: {tgt_shape}")
                else:
                    print(f"Target key not found: {tgt_key_specific}")

                if src_shape != tgt_shape:
                    print(f"Shape mismatch for {src_key_specific} -> {tgt_key_specific}: {src_shape} vs {tgt_shape}")
                else:
                    print(f"Shapes match for {src_key_specific} -> {tgt_key_specific}")

            except Exception as e:
                print(f"Error checking shapes for {src_key_specific} -> {tgt_key_specific}: {e}")

        return io.apply_transforms(source, target, mapping=mapping, transforms=[_import_qkv, _import_qkv_bias, _import_linear_fc1])

   
    @property
    def tokenizer(self):
        #from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.save_hf_tokenizer_assets(str(self)), trust_remote_code=True)

    @property
    def config(self) -> Phi3Config:
        from transformers import Phi3Config as HFPhi3Config

        source = HFPhi3Config.from_pretrained(str(self))

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        output = Phi3Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=False,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output

@io.model_exporter(Phi3Model, "hf")
class HFPhi3Exporter(io.ModelConnector[Phi3Model, "Phi3ForCausalLM"]):
    def init(self) -> "Phi3ForCausalLM":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_config(self.config)

    def apply(self, output_path: Path) -> Path:
        target = self.init()
        source, _ = self.nemo_load(str(self))
        target = self.convert_state(source, target)

        target.cpu().save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        # Convert source weights to target dtype if needed
        for name, param in source.state_dict().items():
            if param.dtype != target.state_dict()[name].dtype:
                param.data = param.data.to(target.state_dict()[name].dtype)

        return io.apply_transforms(source, target, mapping=mapping)

    @property
    def tokenizer(self):
        return io.load_context(str(self)).model.tokenizer.tokenizer

    @property
    def config(self) -> "HFPhi3Config":
        source: Phi3Config = io.load_context(str(self)).model.config

        from transformers import Phi3Config as HFPhi3Config

        return HFPhi3Config(
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            max_position_embeddings=source.seq_length,
            initializer_range=0.02,
            rms_norm_eps=1e-05,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
        )
    

@io.state_transform(
   # source_key="model.layers.*.self_attn.qkv_proj.weight",
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
        qkv_bias = torch.cat((qkv_bias, k[i : i + 1, :, :]))
        qkv_bias = torch.cat((qkv_bias, v[i : i + 1, :, :]))
    # assert qkv_weights.ndim == 3, qkv_weights.shape
    # assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    # assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    # assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_bias = qkv_bias.reshape(
        [
            head_size * (head_num + 2 * num_query_groups),
        ]
    )
    return qkv_bias

    

@io.state_transform(
    # source_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"), # phi-3-mini-4k-instruct
    source_key = ("model.layers.*.mlp.up_proj.weight", "model.layers.*.mlp.linear_fc1.weight"),  # phi-3-small-8k-instruct
    target_key="decoder.layers.*.mlp.linear_fc1.weight",
)
def _import_linear_fc1(down, gate):
    return torch.cat((down, gate), axis=0)


__all__ = [
    "Phi3Config",
    "Phi3Config4B",
    "Phi3Config8B",
    "Phi3Model"
]
