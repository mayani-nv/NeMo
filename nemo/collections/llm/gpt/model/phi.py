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

@dataclass
class Phi3Config(GPTConfig):
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    num_layers: int = 32
    hidden_size: int = 3072
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 32
    num_query_groups: int = 32
    rotary_base: float = 10000.0
    vocab_size: int = 32064

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
class HFPhi3Importer(io.ModelConnector["Phi3ForCausalLM", Phi3Model]):
    def init(self) -> Phi3Model:
        return Phi3Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        from transformers import Phi3ForCausalLM

        # Check if the source is valid model identifier or path
        try:
            source = Phi3ForCausalLM.from_pretrained(str(self), torch_dtype='auto')
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
            "model.embed_tokens.weight": "module.embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "module.decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.self_attn.qkv_proj.weight": "module.decoder.layers.*.self_attention.linear_qkv.weight",
            "model.layers.*.mlp.gate_up_proj.weight": "module.decoder.layers.*.mlp.linear_fc1.weight",
            "model.layers.*.mlp.down_proj.weight": "module.decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "module.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.post_attention_layernorm.weight": "module.decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.norm.weight": "module.decoder.final_layernorm.weight",
            "model.lm_head.weight": "module.output_layer.weight",
        }

        # Print keys for debugging
        source_keys_layers_0 = [key for key in source.state_dict().keys() if key.startswith("model.layers.0.")]
        target_keys_layers_0 = [key for key in target.state_dict().keys() if key.startswith("module.decoder.layers.0.")]
        print("Source state dict keys:", source_keys_layers_0)
        print("Target state dict keys:", target_keys_layers_0)

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

        return io.apply_transforms(source, target, mapping=mapping)


    @property
    def tokenizer(self):
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    @property
    def config(self) -> Phi3Config:
        from transformers import Phi3Config as HFPhi3Config

        source = HFPhi3Config.from_pretrained(str(self))

        output = Phi3Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            num_attention_heads=source.num_attention_heads,
            rotary_base=source.rope_theta,
            share_embeddings_and_output_weights=False,
            params_dtype=torch.bfloat16 if source.torch_dtype == 'bfloat16' else torch.float16,
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
    
    __all__ = [
        "Phi3Config",
        "Phi3Model"
    ]
