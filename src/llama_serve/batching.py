import asyncio
from typing import Callable

class BatchQueue:
    def __init__(
        self,
        loop,
        max_batch_size: int,
        batch_wait_timeout_s: int,
        handle_batch_func: Callable,
    ):
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size

        self._loop = loop
        self._requests_available_event = asyncio.Event()

        self._handle_batch_task = None

        if handle_batch_func is not None:
            self._handle_batch_task = self._loop.create_task(
                self._process_batches(handle_batch_func)
            )
        
    def put(self, request):
        self.queue.put_nowait(request)
        self._requests_available_event.set()
    
    