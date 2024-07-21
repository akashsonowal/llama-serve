import logging
import asyncio
from time import perf_counter
from typing import Type
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from batch_handler import BatchHandler
from inference import LLMInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    prompt: str
    new_tokens: int


class InferenceResponse(BaseModel):
    response: str


class LLamaServe:
    def __init__(
        self,
        model_name: str,
        max_batch_size: int = 128,
        input_schema: Type[BaseModel] = InferenceRequest,
        response_schema: Type[BaseModel] = InferenceResponse,
    ):
        self._app = FastAPI(lifespan=self.lifespan, title="LLamaServe", docs_url="/")
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._app.add_api_route("/health", self.health, methods=["GET"])
        self._app.add_api_route(
            "/endpoint", self.api, methods=["POST"], response_model=response_schema
        )

        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.input_schema = input_schema
        self.response_schema = response_schema

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        app.llm_inf = LLMInference(self.model_name)
        app.batch_handler = BatchHandler(
            max_batch_size=self.max_batch_size, callback_fn=app.llm_inf.generate
        )
        asyncio.create_task(app.batch_handler.consume())
        yield

    async def health(self):
        return Response(status_code=200)

    async def api(self, request: InferenceRequest):
        start = perf_counter()
        try:
            result = await self._app.batch_handler.process_request(request.dict())
            logger.info(f"Done in {(perf_counter() - start):.2f} secs")
            return self.response_schema(response=result)
        except Exception as e:
            logger.exception("Error processing request")
            raise HTTPException(status_code=500, detail="Internal Server Error")

    def run_server(self):
        uvicorn.run(self._app, host="0.0.0.0", port=7000, log_level="info")


if __name__ == "__main__":
    llama_serve = LLamaServe(model_name="/home/ubuntu/llama-serve/artifacts/gpt2")
    llama_serve.run_server()
