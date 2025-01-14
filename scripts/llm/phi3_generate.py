
import os

import torch
import torch.distributed
from megatron.core.inference.common_inference_params import CommonInferenceParams

import nemo.lightning as nl
from nemo.collections.llm import api

if __name__ == "__main__":
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    )
    prompts = [
        "Hello, how are you?",
        "How many r's are in the word 'strawberry'?",
        "Which number is bigger? 10.119 or 10.19?",
    ]
    results = api.generate(
        path=os.path.join(os.environ["NEMO_HOME"], "models", "microsoft/Phi-3-mini-4k-instruct"),
        prompts=prompts,
        trainer=trainer,
        inference_params=CommonInferenceParams(temperature=0.1, top_k=10, num_tokens_to_generate=512),
        text_only=True,
    )
    if torch.distributed.get_rank() == 0:
        for i, r in enumerate(results):
            print(prompts[i])
            print("*" * 50)
            print(r)
            print("\n\n")
