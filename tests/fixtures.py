"""Synthetic CPU engine, tokenizer and head for API behavior tests."""

import asyncio
import base64
import hashlib
import io
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace as NS

import torch
from PIL import Image

from fuzev.contracts import Primitive
from fuzev.encoding import IMAGE_PLACEHOLDER as PLACEHOLDER, VERSION
from fuzev.service_contracts import GenerationRequest, ServiceDecisionRequest
from fuzev_vllm.images import VllmImages
from fuzev_vllm.server import create_app
from fuzev_vllm.service import VllmAPI, VllmService, VllmServiceConfig


class Tokenizer:
    markers = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
    marker_ids = (150001, 150002, 150003)

    def __init__(self):
        self.calls, self.renders = [], []

    def get_vocab(self):
        return {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4}

    def encode(self, text, **kwargs):
        self.calls.append(text)
        if len(text) == 2 and text[0] == " " and "A" <= text[1] <= "Z":
            return [200 + ord(text[1]) - ord("A")]
        tokens = []
        while text:
            for marker, token in zip(self.markers, self.marker_ids):
                if text.startswith(marker):
                    tokens.append(token)
                    text = text[len(marker):]
                    break
            else:
                tokens.append(ord(text[0]) + 1)
                text = text[1:]
        return tokens

    def apply_chat_template(self, messages, **kwargs):
        self.renders.append(deepcopy((messages, kwargs)))
        assert kwargs["tokenize"] is False
        def content(value):
            if isinstance(value, str):
                return value
            return "".join(PLACEHOLDER if part["type"] == "image" else part["text"] for part in value)
        return "[context]" + "|".join(content(item["content"]) for item in messages) + (
            "[assistant]" if kwargs["add_generation_prompt"] else "")

    def decode(self, tokens, **_kwargs):
        pieces = {700: "fixture reasoning", 701: "fixture answer", 2: "</think>", 0: ""}
        return "".join(pieces.get(token, chr(max(0, token - 1))) for token in tokens)


class API:
    reasoning_parser_class = staticmethod(VllmAPI.reasoning_parser_class)
    reasoning_parser = staticmethod(VllmAPI.reasoning_parser)
    @staticmethod
    def image_placeholder(config):
        return PLACEHOLDER

    def sampling(self, request, defaults, available_tokens, parser=None):
        self.sampling_args = request, deepcopy(defaults), available_tokens
        return NS(kind="generate", request=request)

    def pooling(self, labels=None):
        return NS(task="fuzev_features", extra_kwargs=(
            None if labels is None else {"baseline_token_ids": list(labels)}))

    def reasoning(self, parser_class, tokenizer, text, request, kwargs, prompt_tokens):
        self.reasoning_args = parser_class, tokenizer, text, request, kwargs, prompt_tokens
        return "fixture reasoning", "fixture answer"


class Engine:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.renderer = Renderer()
        self.calls, self.inputs, self.aborted = [], [], []
        self.active = 0
        self.wait, self.abort_wait = None, None
        self.stream_wait = None
        self.entered, self.aborting = asyncio.Event(), asyncio.Event()
        self.completion = [700, 701, 0]
        self.text = "fixture reasoning</think>fixture answer"
        self.patch_output = lambda output, kind: None
        self.tasks = ("generate", "fuzev_features")
        self.failure = None
        self.shutdown_called = False
        self.vllm_config = NS(model_config=NS(
            get_vocab_size=lambda: 200000, max_model_len=4096, hf_text_config=NS(hidden_size=8),
            get_diff_sampling_param=lambda: {"temperature": .4, "top_p": .8}))

    async def get_supported_tasks(self):
        return self.tasks

    def get_tokenizer(self):
        return self.tokenizer

    async def check_health(self):
        pass

    def shutdown(self):
        self.shutdown_called = True

    async def abort(self, request_id):
        self.aborted.append(request_id)
        self.aborting.set()
        if self.abort_wait:
            await self.abort_wait.wait()

    async def _run(self, kind, prompt, params, request_id):
        tokens = tuple(prompt["prompt_token_ids"])
        self.calls.append((kind, tokens, params, request_id))
        self.inputs.append(deepcopy(prompt))
        self.active += 1
        self.entered.set()
        try:
            if self.wait:
                await self.wait.wait()
            await asyncio.sleep(0)
            if self.failure is not None:
                raise self.failure
            if kind == "encode":
                labels = (params.extra_kwargs or {}).get("baseline_token_ids", [])
                data = torch.full((8,), float(sum(tokens) % 127))
                if labels:
                    data = torch.cat((data, torch.tensor(labels, dtype=torch.float32) / 4))
                payload = NS(data=data)
            else:
                payload = [NS(index=0, token_ids=list(self.completion), text=self.text, finish_reason="stop")]
                if params.request.stream:
                    # Streams use delta output: the final chunk carries only the remaining tokens.
                    payload[0].token_ids, payload[0].text = [2, 701, 0], "</think>fixture answer"
                    yield NS(request_id=request_id, outputs=[NS(index=0,
                        token_ids=[700], text="fixture reasoning", finish_reason=None)],
                        prompt_token_ids=list(tokens), finished=False, num_cached_tokens=1)
                    await asyncio.sleep(0)
                    if self.stream_wait is not None:
                        await self.stream_wait.wait()
            output = NS(request_id=request_id, outputs=payload, prompt_token_ids=list(tokens),
                        finished=True, num_cached_tokens=1)
            self.patch_output(output, kind)
            yield output
        finally:
            self.active -= 1

    def generate(self, prompt, params, request_id):
        return self._run("generate", prompt, params, request_id)

    def encode(self, prompt, params, request_id):
        return self._run("encode", prompt, params, request_id)


@dataclass(frozen=True)
class Position:
    offset: int
    length: int
    is_embed: object = None


@dataclass
class Elem:
    data: object


class Renderer:
    def __init__(self, limit=4):
        self.mm_processor = NS(info=NS(allowed_mm_limits={"image": limit}))
        self.calls, self.pil_images = [], []
        self.wait = None
        self.entered = asyncio.Event()

    async def process_for_engine_async(self, prompt, arrival_time, *, skip_mm_cache):
        assert skip_mm_cache and prompt["mm_processor_kwargs"] == {"truncation": False}
        image, = prompt["multi_modal_data"]["image"]
        assert image.mode == "RGB"
        raw = tuple(prompt["prompt_token_ids"])
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        self.calls.append((raw, digest))
        self.pil_images.append(image)
        self.entered.set()
        if self.wait:
            await self.wait.wait()
        pad = raw.index(150002)
        return {"type": "multimodal", "prompt_token_ids": list(raw[:pad] + (150002,) * 4 + raw[pad + 1:]),
            "mm_kwargs": {"image": [{"pixel_values": Elem(torch.full((4, 3), float(image.getpixel((0, 0))[0])))}]},
            "mm_hashes": {"image": [digest]}, "mm_placeholders": {"image": [Position(pad, 4)]}}


class FixtureHead:
    head_id = "reference"
    max_options = 16
    primitives = tuple(Primitive)
    encoding_version = VERSION
    calibration_id = "raw"
    metadata = {"hidden_size": 8, "release_date": "2026-10-12"}

    def __init__(self):
        self.closed = False
        self.observed = []

    def score(self, primitive, query, candidates, *, baseline):
        self.observed.append((primitive, query.clone(), candidates.clone(), baseline.clone()))
        assert query.shape == (8,) and candidates.shape[1] == 8
        probabilities = torch.softmax(baseline, 0).tolist()
        return baseline.tolist(), probabilities, probabilities[:]

    def close(self):
        self.closed = True


def config(**kwargs):
    return VllmServiceConfig(**({"model_name": "fixture-only", "head_directory": "/fixture/not-opened"} | kwargs))


def generation(**kwargs):
    return GenerationRequest(**({"model": "fixture-only", "prompt": "prompt-A", "max_tokens": 8} | kwargs))


def decision(**kwargs):
    return ServiceDecisionRequest(**({"model": "fixture-only", "primitive": "choice", "question": "Which?",
        "messages": [{"role": "user", "content": "context"}],
        "candidates": [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]} | kwargs))


def image_string(color=(10, 20, 30), *, data_url=False, format="PNG"):
    stream = io.BytesIO()
    with Image.new("RGB", (8, 8), color) as image:
        image.save(stream, format=format)
    encoded = base64.b64encode(stream.getvalue()).decode()
    return ("data:image/" + format.lower() + ";base64," if data_url else "") + encoded


def service(**kwargs):
    engine = Engine()
    return VllmService(engine, engine.tokenizer, FixtureHead(), config(**kwargs), api=API(),
                      images=VllmImages(engine.renderer, PLACEHOLDER))


def application(*, head_max_options=16, api_key=None, max_request_bytes=None, **kwargs):
    cfg = config(**kwargs)
    holder = NS()

    @asynccontextmanager
    async def context():
        holder.engine = Engine()
        try:
            yield holder.engine
        finally:
            holder.context_exited = True

    async def builder(engine, cfg):
        head = FixtureHead() if cfg.head_directory else None
        if head is not None:
            head.max_options = head_max_options
        holder.service = VllmService(engine, engine.tokenizer,
            head, cfg, api=API(), images=VllmImages(engine.renderer, PLACEHOLDER))
        return holder.service

    holder.app = create_app(context, cfg, builder=builder, api_key=api_key,
                            max_request_bytes=max_request_bytes)
    return holder


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.005)
