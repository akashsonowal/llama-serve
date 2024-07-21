import copy
import time
import torch
import random
from uuid import uuid4
from typing import List
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMInference:
    def __init__(self, model_name) -> None:
        self._init_model(model_name)

    def _init_model(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

    def _init_batch(self, requests: List[dict]):
        uids = [r["uid"] for r in requests]
        prompts = [r["prompt"] for r in requests]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        attention_mask = inputs["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)  # all pad tokens are filled with 1
        return {
            "position_ids": position_ids,
            "responses": copy.copy(prompts),  # dummy placeholder
            "tokens_remaining": [r["new_tokens"] for r in requests],  # new tokens
            **inputs,
        }, uids

    def generate_next_token(self, batch):
        inputs = copy.copy(batch)
        inputs.pop("responses")
        inputs.pop("tokens_remaining")
        next_token_ids, past_key_values = self.generate_batch_tokens_with_past(inputs)
        next_tokens = self.tokenizer.batch_decode(next_token_ids)
        return self.get_next_inputs(batch, next_token_ids, past_key_values, next_tokens)

    def generate_batch_tokens_with_past(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        last_logits = logits[:, -1, :]
        next_token_ids = last_logits.argmax(dim=1)
        return next_token_ids, outputs.past_key_values

    def get_next_inputs(self, batch, next_token_ids, past_key_values, next_tokens):
        return {
            # '-1' here means the remaining elements for this dim
            "input_ids": next_token_ids.reshape((-1, 1)).to(self.device),
            # increment last, discard the rest
            "position_ids": batch["position_ids"][:, -1].unsqueeze(-1) + 1,
            # concatenate vector of 1's with shape [batch_size]
            "attention_mask": torch.cat(
                [
                    batch["attention_mask"],
                    torch.ones((next_token_ids.shape[0], 1), device=self.device),
                ],
                dim=1,
            ),
            "past_key_values": past_key_values,
            "responses": [r1 + r2 for r1, r2 in zip(batch["responses"], next_tokens)],
            "tokens_remaining": [v - 1 for v in batch["tokens_remaining"]],
        }

    def merge_batches(self, batch1, batch2, uids1, uids2):
        attn_mask1 = batch1["attention_mask"]
        attn_mask2 = batch2["attention_mask"]
        max_seq_len = max(attn_mask1.shape[1], attn_mask2.shape[1])

        padding1 = max_seq_len - attn_mask1.shape[1]
        padding2 = max_seq_len - attn_mask2.shape[1]
        attn_mask1 = F.pad(attn_mask1, (padding1, 0), "constant", 0).to(self.device)
        attn_mask2 = F.pad(attn_mask2, (padding2, 0), "constant", 0).to(self.device)

        past_kv1 = batch1["past_key_values"]
        past_kv2 = batch2["past_key_values"]

        padded_kv1 = []
        for i in range(len(past_kv1)):
            k, v = past_kv1[i]
            k = F.pad(k, (0, 0, padding1, 0), "constant", 0).to(self.device)
            v = F.pad(v, (0, 0, padding1, 0), "constant", 0).to(self.device)
            padded_kv1.append((k, v))

        padded_kv2 = []
        for i in range(len(past_kv2)):
            k, v = past_kv2[i]
            k = F.pad(k, (0, 0, padding2, 0), "constant", 0).to(self.device)
            v = F.pad(v, (0, 0, padding2, 0), "constant", 0).to(self.device)
            padded_kv2.append((k, v))

        input_ids = torch.cat([batch1["input_ids"], batch2["input_ids"]], dim=0).to(self.device)
        position_ids = torch.cat([batch1["position_ids"], batch2["position_ids"]], dim=0).to(self.device)
        attn_mask = torch.cat([attn_mask1, attn_mask2], dim=0).to(self.device)

        past_kv = []
        for i in range(len(padded_kv1)):
            k1, v1 = padded_kv1[i]
            k2, v2 = padded_kv2[i]
            k = torch.cat([k1, k2], dim=0).to(self.device)
            v = torch.cat([v1, v2], dim=0).to(self.device)
            past_kv.append((k, v))

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attn_mask,
            "past_key_values": past_kv,
            "responses": batch1["responses"] + batch2["responses"],
            "tokens_remaining": batch1["tokens_remaining"] + batch2["tokens_remaining"],
        }, uids1 + uids2

    def filter_batch(self, batch, uids):
        remove_indices = []
        for i, tokens_remaining in enumerate(batch["tokens_remaining"]):
            if tokens_remaining <= 0:
                remove_indices.append(i)

        batch_size = batch["input_ids"].size(0)
        mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        mask[remove_indices] = False

        input_ids = batch["input_ids"][mask]
        position_ids = batch["position_ids"][mask]
        attention_mask = batch["attention_mask"][mask]

        remaining_uids = [r for i, r in enumerate(uids) if i not in remove_indices]
        removed_uids = [r for i, r in enumerate(uids) if i in remove_indices]

        responses = [r for i, r in enumerate(batch["responses"]) if i not in remove_indices]
        removed_responses = [r for i, r in enumerate(batch["responses"]) if i in remove_indices]

        tokens_remaining = [v for i, v in enumerate(batch["tokens_remaining"]) if i not in remove_indices]

        past_key_values = batch["past_key_values"]
        new_past_key_values = []

        for i in range(len(past_key_values)):
            k, v = past_key_values[i]
            k = k[mask]
            v = v[mask]
            new_past_key_values.append((k, v))

        past_key_values = new_past_key_values
        if input_ids.size(0) > 0:
            zero_mask = attention_mask == 0
            cumprod = zero_mask.cumprod(dim=1)
            leading_zeros_count = cumprod.sum(dim=1)
            min_leading_zeros = torch.min(leading_zeros_count)
            truncation_offset = min_leading_zeros.item()

            attention_mask = attention_mask[:, truncation_offset:]
            past_key_values = past_key_values
            new_past_key_values = []
            for i in range(len(past_key_values)):
                k, v = past_key_values[i]
                k = k[:, :, truncation_offset:, :]
                v = v[:, :, truncation_offset:, :]
                new_past_key_values.append((k, v))
            past_key_values = new_past_key_values

        return (
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "responses": responses,
                "tokens_remaining": tokens_remaining,
            },
            remaining_uids,
            removed_uids,
            removed_responses,
        )

    def generate(self, request_queue, response_queue, batch_size):
        if len(request_queue) == 0:
            return
        
        batch, uids = self._init_batch(request_queue[:batch_size])
        cached_batch = self.generate_next_token(batch)  # prefill step
        request_queue = request_queue[batch_size:]

        while len(request_queue) > 0 or cached_batch["input_ids"].size(0) > 0:
            batch_capacity = batch_size - cached_batch["input_ids"].size(0)

            if batch_capacity > 0 and len(request_queue) > 0:
                new_batch, new_uids = self._init_batch(request_queue[:batch_capacity])
                new_batch = self.generate_next_token(new_batch)
                request_queue = request_queue[batch_capacity:]
                cached_batch, uids = self.merge_batches(
                    cached_batch, new_batch, uids, new_uids
                )

            cached_batch = self.generate_next_token(cached_batch)
            cached_batch, uids, finished_uids, finished_responses = self.filter_batch(
                cached_batch, uids
            )

            for uid, response in zip(finished_uids, finished_responses):
                response_queue[uid] = response


if __name__ == "__main__":
    queue_size = 32
    batch_size = 8

    prompts = [
        "The quick brown fox jumped over the",
        "The rain in Spain falls",
        "What comes up must",
    ]

    random.seed(42)
    request_queue = [
        {
            "uid": str(uuid4()),
            "prompt": random.choice(prompts),
            "new_tokens": 100 if i % batch_size == 0 else 10,
        }
        for i in range(queue_size)
    ]

    model_name = "/home/ubuntu/llama-serve/artifacts/gpt2"
    llm_inf = LLMInference(model_name)
    responses = {}
    t0 = time.time()
    print(len(request_queue))
    llm_inf.generate(request_queue, responses, batch_size)

    duration_s = time.time() - t0
    print("duration", duration_s)
    print(len(responses))