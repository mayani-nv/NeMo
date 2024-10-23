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
from typing import TYPE_CHECKING, Annotated, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel
from nemo.collections.llm.utils import Config
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from transformers import Phi3Model, Phi3Config
    from transformers import LlamaForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

@dataclass
class Phi3Config(GPTConfig):
    vocab_size: int = 32064
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attention_dropout: float = 0.0
    hidden_act: str = 'silu'
    max_position_embeddings: int = 4096
    original_max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    bos_token_id: int = 1
    eos_token_id: int = 32000
    pad_token_id: int = 32000

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
        return Phi3Model(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path):
        from transformers import AutoModelForCausalLM

        source_model = AutoModelForCausalLM.from_pretrained(str(self), torch_dtype='auto')
        target_model = self.init()
        trainer = self.nemo_setup(target_model)
        self.convert_state(source_model, target_model)
        self.nemo_save(output_path, trainer)

        print(f"Converted Phi-3 model to NeMo format, saved to {output_path}.")

        teardown(trainer, target_model)
        del trainer, target_model

        return output_path

__all__ = [
    "Phi3Config",
    "Phi3Model",
]
