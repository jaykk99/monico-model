"""
Monico Inference Server — OpenAI-compatible API

monico-agent calls this endpoint as the default model.
Supports:
  - /v1/chat/completions  (streaming + non-streaming)
  - /v1/models
  - /v1/completions

Run:
  python -m src.serve.inference_server --model checkpoints/monico-7b-sft --port 8080
Or GGUF (local, no GPU needed):
  python -m src.serve.inference_server --gguf monico-7b-Q4_K_M.gguf --port 8080
"""
import time, json, uuid, logging, argparse
from typing import Optional, List, Dict, AsyncGenerator
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

log = logging.getLogger("monico.serve")
app = FastAPI(title="Monico Inference API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global model handle (set at startup)
MODEL = None
TOKENIZER = None
MODEL_NAME = "monico-7b"

# ── Schemas ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "monico"
    messages: List[Message]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class CompletionRequest(BaseModel):
    model: str = "monico"
    prompt: str
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{
        "id": MODEL_NAME, "object": "model",
        "created": 1700000000, "owned_by": "monico",
    }]}

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    if req.stream:
        return StreamingResponse(
            stream_tokens(prompt, req.max_tokens, req.temperature, req.top_p, req.stop),
            media_type="text/event-stream"
        )

    output = generate(prompt, req.max_tokens, req.temperature, req.top_p, req.stop)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": output},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(output.split()), "total_tokens": 0},
    }

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    output = generate(req.prompt, req.max_tokens, req.temperature)
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"text": output, "index": 0, "finish_reason": "stop"}],
    }

# ── Generation helpers ─────────────────────────────────────────────────────────

def generate(prompt: str, max_tokens: int, temperature: float, top_p: float = 0.9, stop=None) -> str:
    import torch
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        out = MODEL.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    text = TOKENIZER.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if stop:
        for s in stop:
            text = text.split(s)[0]
    return text.strip()

async def stream_tokens(prompt, max_tokens, temperature, top_p, stop) -> AsyncGenerator[str, None]:
    """SSE streaming generator — yields OpenAI-format data: chunks."""
    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread

    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, max_new_tokens=max_tokens, temperature=temperature,
                      top_p=top_p, do_sample=temperature > 0, streamer=streamer)
    t = Thread(target=MODEL.generate, kwargs=gen_kwargs)
    t.start()

    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    for chunk in streamer:
        data = json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                           "model": MODEL_NAME, "choices": [{"delta": {"content": chunk}, "index": 0}]})
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"

# ── Startup ────────────────────────────────────────────────────────────────────

def load_hf_model(path: str):
    global MODEL, TOKENIZER, MODEL_NAME
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log.info(f"Loading HF model from {path} ...")
    TOKENIZER = AutoTokenizer.from_pretrained(path)
    MODEL = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto",
                                                  attn_implementation="flash_attention_2")
    MODEL_NAME = path.split("/")[-1]
    log.info("Model loaded.")

def load_gguf_model(gguf_path: str):
    global MODEL, TOKENIZER, MODEL_NAME
    from llama_cpp import Llama
    log.info(f"Loading GGUF model from {gguf_path} ...")
    MODEL = Llama(model_path=gguf_path, n_ctx=32768, n_gpu_layers=-1, verbose=False)
    MODEL_NAME = gguf_path.split("/")[-1].replace(".gguf", "")
    log.info("GGUF model loaded.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=None, help="HF model path")
    p.add_argument("--gguf",   default=None, help="GGUF file path (llama.cpp)")
    p.add_argument("--port",   type=int, default=8080)
    p.add_argument("--host",   default="0.0.0.0")
    args = p.parse_args()

    if args.gguf:
        load_gguf_model(args.gguf)
    elif args.model:
        load_hf_model(args.model)
    else:
        log.warning("No model specified — server will start but /chat won't work")

    uvicorn.run(app, host=args.host, port=args.port)
