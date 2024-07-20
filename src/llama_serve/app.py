import logging
import asyncio

from typing import Callable, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import Config, Server

# from .batching import BatchQueue

logger = logging.getLogger(__name__)


class MLServe:
    def __init__(
        self,
        handle: Callable,
        input_schema: Optional[BaseModel],
        response_schema: Optional[BaseModel],
        max_batch_size: int = 16,
#         batch_wait_timeout_s: int = 0.05,
    ):
        self.loop = asyncio.get_event_loop()
#         self.queue = BatchQueue(self.loop, max_batch_size, batch_wait_timeout_s, handle)
#         self.app = FastAPI(title="MLServe: A scalable ML Server", docs_url="/")
#         self.config = Config(app=self.app, loop=self.loop, host="0.0.0.0", port=8080)
#         self.server = Server(self.config)

#         INPUT_SCHEMA = input_schema
#         RESPONSE_SCHEMA = response_schema

        def health():
            return Response(status_code=200)

#         async def api(request: INPUT_SCHEMA) -> RESPONSE_SCHEMA:
#             try:
#                 return await self.queue.process_request(request)
#             except Exception as e:
#                 logger.exception("Error processing request")
#                 raise HTTPException(status_code=500, detail="Internal Server Error")

        self.app.add_api_route("/health", health, methods=["GET"])
#         self.app.add_api_route("/endpoint", api, methods=["POST"])

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def run_server(self):
        self.loop.run_until_complete(self.server.serve())

if __name__ == "__main__":
    def f(x):
        return x
    ml_serve = MLServe(f)
    ml_serve.run_server()