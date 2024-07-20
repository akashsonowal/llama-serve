import time
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:
    def __init__(self, model_name, quantize=False) -> None:
        self._setup_model(model_name, quantize)
        self._bs = 8
    
    def _setup_model(self, model_name, quantize):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def _init_batch(self):
        "prefill"
        pass 

    def merge_batches(self):
        pass 

    def filter_batch(self):
        pass 

    def generate_next_token(self, batch):
        pass 

    def generate_reponse(self, prompts):
        pass 

    
    @torch.no_grad()
    def generate(self, requests):
        request_queue = copy.copy(requests)
        responses = [None] * len(requests)

        batch = self._init_batch(request_queue[:self._bs])

        cached_batch = self.generate_next_token(batch)
        request_queue = request_queue[self._bs:]

        # continue until both the request queue is fully drained and every input
        # within the cached_batch has completed generation

        while len(request_queue) > 0 or cached_batch["input_ids"].size(0) > 0:
            batch_capacity = self._bs - cached_batch["input_ids"].size(0)

            if batch_capacity > 0 and len(request_queue) > 0:
                # prefill
                new_batch = self.init_batch(request_queue[:batch_capacity])
                new_batch = self.generate_next_token(new_batch)
                request_queue = request_queue[batch_capacity:]

                cached_batch = self.merge_batches(cached_batch, new_batch)
            
            # decode
            cached_batch = self.generate_next_token(cached_batch)

            # remove any inputs that have finished generation
            cached_batch, removed_indices, completed_responses = self.filter_batch(cached_batch)

            for idx, resp in zip(removed_indices, completed_responses):
                responses[idx] = resp
        return responses