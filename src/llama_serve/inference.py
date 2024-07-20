import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:
    def __init__(self, model_name, quantize=False) -> None:
        self._setup_model(model_name, quantize)
    
    def _setup_model(self, model_name, quantize):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    @torch.no_grad()
    def generate(self, prompt):
        pass 