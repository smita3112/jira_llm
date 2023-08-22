from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import logging

logging.basicConfig(
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

model_name = "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2"

logger.info("Initiating tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    trust_remote_code=True,
    padding_side="left",
)

logger.info("Initiating model")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
logger.info("Setting model to evaluation (inference) mode")
model.eval()

app = FastAPI()


class LLMRequest(BaseModel):
    prompt: str
    temperature: float = 0.1
    min_new_tokens: int = 2
    max_new_tokens: int = 1024
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1.2
    renormalize_logits: bool = True


class LLMResponse(BaseModel):
    response: str


def generate_response_stream(request: LLMRequest):
    formatted_prompt = f"<|prompt|>{request.prompt}<|endoftext|><|answer|>"
    logger.debug(f"Formatted prompt: \n{formatted_prompt}")
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False
    ).to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        min_new_tokens=request.min_new_tokens,
        max_new_tokens=request.max_new_tokens,
        do_sample=request.do_sample,
        num_beams=request.num_beams,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        renormalize_logits=request.renormalize_logits
    )
    thread = Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )
    thread.start()
    for new_text in streamer:
        yield new_text


@app.get("/")
def get_model_card():
    return {
        "model_name": model_name,
        "hugging_face_model_card": "https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2"
    }


@app.post("/generate")
def generate(request: LLMRequest) -> LLMResponse:
    logger.info("Generating")
    generated_text = ""
    for new_text in generate_response_stream(request):
        generated_text += new_text
    logger.info("Returning response")
    return LLMResponse(response=generated_text)


@app.post("/generate_stream")
def generate_stream(request: LLMRequest) -> StreamingResponse:
    logger.info("Generating stream")
    return StreamingResponse(generate_response_stream(request))
