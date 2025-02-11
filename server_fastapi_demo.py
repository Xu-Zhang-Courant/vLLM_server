"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import time
import argparse
import asyncio
import json
import ssl
from argparse import Namespace
from typing import Any, AsyncGenerator, Optional
import random
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import fastapi
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (random_uuid)
import string
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import keyboard
import threading
import sys
import signal
import os

set_seed(42)

running_time_result = []
app = FastAPI()

sampling_params = SamplingParams(use_beam_search = False, temperature=0.01, top_p = 1, top_k = 1, max_tokens = 1000, min_tokens = 0, repetition_penalty = 1, length_penalty = 1, seed = 42)

tokenizer = AutoTokenizer.from_pretrained("/root/model-tokenizer-location", trust_remote_code=True)

# id Generator. Use seed to ensure every time the server get different id.
def get_task_id(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    start_time = time.time()

    # request pre-processing
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    posfix = "<|im_end|>\n<|im_start|>assistant"
    prompt = prefix + prompt + posfix
    prefix_pos = request_dict.pop("prefix_pos", None)
    stream = request_dict.pop("stream", False)
    global sampling_params
    sampling_params = sampling_params
    task_id = random_uuid()
    idx = request_dict.pop("idx")

    # actual processing of the request
    results_generator = engine.generate(prompt,
                                        sampling_params,
                                        task_id)
    
    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(task_id)
            return Response(status_code=499)
        final_output = request_output
    
    assert final_output is not None

    text_outputs_list = [output.text for output in final_output.outputs]
    text_outputs = text_outputs_list[0]
    end_time = time.time()
    result = result_wrapper(text_outputs, start_time, end_time, idx)

    # Output the result
    return JSONResponse(result)

# pack all stat and output into json
def result_wrapper(response_text, start_time, end_time, idx):
    total_time = end_time - start_time
    output_token_count = len(tokenizer.tokenize(response_text))
    output_speed = output_token_count / total_time
    speed_result = f"[{total_time:.2f}s/it, est. start_time: at {start_time:.2f} , output: {output_speed:.2f} toks/s]"    
    task_id = get_task_id()
    result = {
    "response": response_text,
    "total_time": total_time,
    "start_time": start_time,
    "end_time": end_time,
    "output_speed": output_speed,
    "length_of_output": len(response_text),
    "speed_result": speed_result,
    "task_id": task_id,
    "idx": idx
    }
    return result

def shutdown():
    save_log_to_json(running_time_result)

    os.kill(os.getpid(), signal.SIGTERM)
    return fastapi.Response(status_code=200, content='Server shutting down...')
    
app.add_api_route('/shutdown', shutdown, methods=['GET'])

def save_log_to_json(running_time_result, filename="/root/test_results/server_log.json"):
    with open(filename, 'w') as json_file:
        json.dump(running_time_result, json_file, indent=4)
    print(f"Log saved to {filename}")

if __name__ == "__main__":
    # Start the FastAPI server in a separate thread
    engine = AsyncLLMEngine.from_engine_args(
    AsyncEngineArgs(
        model="/root/model-weight-location",
        max_model_len=1000,
        engine_use_ray = False,
        worker_use_ray=False,
        trust_remote_code=True
    )
)

    uvicorn.run(app,
                host="127.0.0.1",
                port=6006)

