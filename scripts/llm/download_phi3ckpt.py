from pathlib import Path
from nemo.collections.llm import import_ckpt
from nemo.collections.llm.gpt.model.phi3mini import Phi3ConfigMini, Phi3Model

if __name__ == "__main__":
    import_ckpt(model=Phi3Model(Phi3ConfigMini()),source='hf://microsoft/Phi-3-mini-4k-instruct')
