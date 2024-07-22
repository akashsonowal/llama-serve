import time
import random
import asyncio
from uuid import uuid4
from typing import Awaitable, Callable

from inference import LLMInference

class BatchHandler:
    def __init__(
        self,
        max_batch_size: int,
        callback_fn: Callable[[list, dict, int], None],
    ) -> None:
        self._queue = []
        self._responses = {}
        self.max_batch_size = max_batch_size
        self.callback_fn = callback_fn
    
    async def process_request(self, data: dict) -> Awaitable:
        data["uid"] = str(uuid4())
        self._queue.append(data)
        return await self.wait_for_reply(data["uid"])

    async def consume(self):
        while True:
            self.callback_fn(self._queue, self._responses, self.max_batch_size)
            await asyncio.sleep(0.001)
    
    async def wait_for_reply(self, uid: str) -> dict:
        while True:
            if uid in self._responses:
                response = self._responses[uid]
                del self._responses[uid]
                return response
            await asyncio.sleep(0.001)

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
            "prompt": random.choice(prompts),
            "new_tokens": 100 if i % batch_size == 0 else 10,
        }
        for i in range(queue_size)
    ]

    model_name = "/home/ubuntu/llama-serve/artifacts/gpt2"
    llm_inf = LLMInference(model_name)

    async def main():
        handler = BatchHandler(max_batch_size=32, callback_fn=llm_inf.generate)
        asyncio.create_task(handler.consume())
        t0 = time.time()
        tasks = [handler.process_request(data) for data in request_queue]
        res = await asyncio.gather(*tasks)
        print(f"Processed {len(request_queue)} queries to give {len(res)} responses in {time.time() - t0} seconds")

    asyncio.run(main())