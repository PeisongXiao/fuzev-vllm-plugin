"""Generation and shared decision scoring through one existing AsyncLLM."""

import asyncio
import time
from contextlib import aclosing, asynccontextmanager
from typing import Literal
from uuid import uuid4

import torch
from anyio import create_memory_object_stream
from pydantic import BaseModel, ConfigDict, Field

from fuzev.api import APIError, decision_overflow, validation_error
from fuzev.encoding import IMAGE_PLACEHOLDER, TEMPLATE, VERSION, candidate_suffix, labels, query_suffix
from fuzev.head import ServingHead
from fuzev.prefix import DEFAULT_PREFIX_CAPACITY, PrefixRegistry
from fuzev.service_contracts import GenerationRequest

from .images import MultimodalPayload, VllmImages


class VllmServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_name: str = Field(min_length=1)
    head_directory: str | None = None
    head_device: str | None = "cpu"
    calibration: Literal["raw", "calibrated"] = "raw"
    reasoning_parser: str | None = "qwen3"
    prefix_capacity: int = Field(default=DEFAULT_PREFIX_CAPACITY, ge=1)
    generation_chat_template_kwargs: dict = Field(default_factory=dict)


async def drain_tasks(tasks):
    """Keep request resources alive until cleanup survives repeated cancellation."""
    pending = asyncio.gather(*tasks, return_exceptions=True)
    interrupted = False
    while not pending.done():
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            interrupted = True
    pending.result()
    if interrupted:
        raise asyncio.CancelledError


@asynccontextmanager
async def stream_in_task(iterator):
    """Keep a suspended generator's resources owned by a cancellable task."""
    sender, receiver = create_memory_object_stream(1)

    async def produce():
        try:
            async with aclosing(iterator):
                async for item in iterator:
                    await sender.send(item)
        finally:
            sender.close()

    task = asyncio.create_task(produce())

    async def consume():
        async for item in receiver:
            yield item
        task.result()

    try:
        yield task, consume()
    finally:
        receiver.close()
        if not task.done() and not task.cancelling():
            task.cancel()
        await drain_tasks([task])


class VllmAPI:
    """Deferred engine imports also allow CPU-only API tests."""

    @staticmethod
    def image_placeholder(config):
        from vllm.model_executor.model_loader.utils import get_model_architecture
        model_cls, _ = get_model_architecture(config.model_config)
        return model_cls.get_placeholder_str("image", 1)

    @staticmethod
    def sampling(request, defaults, available_tokens, parser=None):
        from vllm.exceptions import VLLMValidationError
        from vllm.sampling_params import RequestOutputKind, SamplingParams
        options = dict(defaults)
        for name in ("temperature", "top_p", "top_k", "seed", "max_tokens"):
            value = getattr(request, name)
            if value is not None:
                options[name] = value
        if request.stop:
            options["stop"] = list(request.stop)
        if request.max_tokens is None:
            options["max_tokens"] = min(options.get("max_tokens") or available_tokens, available_tokens)
        if request.messages is not None and getattr(parser, "engine_based_streaming", False):
            options["skip_special_tokens"] = False
        try:
            return SamplingParams(**options, output_kind=(
                RequestOutputKind.DELTA if request.stream else RequestOutputKind.FINAL_ONLY))
        except VLLMValidationError as error:
            raise validation_error(error.parameter or "body", str(error)) from None

    @staticmethod
    def pooling(labels=None):
        from vllm.pooling_params import PoolingParams
        return PoolingParams(task="fuzev_features", extra_kwargs=(
            None if labels is None else {"baseline_token_ids": list(labels)}))

    @staticmethod
    def reasoning_parser_class(name):
        from vllm.reasoning import ReasoningParserManager
        return ReasoningParserManager.get_reasoning_parser(name)

    @staticmethod
    def reasoning_parser(parser_class, tokenizer, template_kwargs, prompt_tokens):
        parser = parser_class(tokenizer, chat_template_kwargs=template_kwargs)
        parser.adjust_initial_state_from_prompt(list(prompt_tokens))
        return parser

    @staticmethod
    def reasoning(parser_class, tokenizer, text, request, template_kwargs, prompt_tokens):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        parser = VllmAPI.reasoning_parser(parser_class, tokenizer, template_kwargs, prompt_tokens)
        protocol = ChatCompletionRequest(
            messages=[item.model_dump() for item in request.messages], chat_template_kwargs=template_kwargs)
        return parser.extract_reasoning(text, protocol)


async def build_service(engine_client, config, *, head_factory=ServingHead, api=None):
    required = {"generate", "fuzev_features"} if config.head_directory else {"generate"}
    if not required.issubset(await engine_client.get_supported_tasks()):
        raise ValueError("Decision serving requires the FuZev vLLM fork; see the FuZev install guide's Install The Engine section")
    head = None
    try:
        if config.head_directory:
            head = head_factory(config.head_directory,
                device=config.head_device, calibration=config.calibration)
            width = engine_client.vllm_config.model_config.hf_text_config.hidden_size
            if width != head.metadata["hidden_size"]:
                raise ValueError(f"Head feature width {head.metadata['hidden_size']} does not match model width {width}")
            if head.encoding_version != VERSION:
                raise ValueError(f"Decision head requires encoding {head.encoding_version}; installed encoding is {VERSION}")
        return VllmService(engine_client, engine_client.get_tokenizer(), head, config, api=api)
    except BaseException:
        if head is not None:
            head.close()
        raise


class VllmService:
    def __init__(self, engine, tokenizer, head, config, *, api=None, images=None):
        self.engine, self.tokenizer, self.head = engine, tokenizer, head
        self.config = config.model_copy(deep=True)
        model = engine.vllm_config.model_config
        self.context_window = model.max_model_len
        self.sampling_defaults = model.get_diff_sampling_param()
        self.api = api or VllmAPI()
        self.reasoning_parser_class = (self.api.reasoning_parser_class(config.reasoning_parser)
                                       if config.reasoning_parser else None)
        self.template = TEMPLATE.read_text() if head is not None else None
        self.operations = {}
        self.prefixes = PrefixRegistry(capacity=config.prefix_capacity)
        self.label_ids = self._label_ids(head.max_options) if head is not None else ()
        self.closed = False
        self.images = images
        renderer = getattr(engine, "renderer", None)
        if (images is None and renderer is not None and renderer.mm_processor is not None
                and renderer.mm_processor.info.allowed_mm_limits.get("image", 0) > 0):
            self.images = VllmImages(renderer, self.api.image_placeholder(engine.vllm_config))

    def _check(self):
        if self.closed:
            raise APIError(503, "Server is closing")

    @asynccontextmanager
    async def _operation(self, request_id):
        self._check()
        if request_id in self.operations:
            raise APIError(409, "Request ID is already active")
        self.operations[request_id] = asyncio.current_task()
        try:
            yield
        finally:
            self.operations.pop(request_id, None)

    def release_prefix(self, key):
        return self.prefixes.release(key)

    def _get_prefix(self, key):
        return self.prefixes.get(key)

    def _encode(self, text):
        return tuple(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def _label_ids(self, count):
        encoded = [self._encode(" " + label) for label in labels(count)]
        if (any(len(label) != 1 for label in encoded)
                or len({label[0] for label in encoded}) != count):
            raise RuntimeError("The tokenizer must encode option labels as distinct single tokens")
        return tuple(label[0] for label in encoded)

    def _messages(self, messages, *, generation, image=False, chat_template_kwargs=None):
        from jinja2 import TemplateError
        values = [dict(item) for item in messages]
        if image:
            user = next((item for item in values if item["role"] == "user"), None)
            if user is None:
                user = {"role": "user", "content": "" if generation else "Consider the supplied image."}
                if generation:
                    values.append(user)
                else:
                    values.insert(0, user)
            user["content"] = [{"type": "image"}, {"type": "text", "text": user["content"]}]
        options = dict(chat_template_kwargs or {}) if generation else dict(
            chat_template=self.template, enable_thinking=False, reasoning_effort="none", preserve_thinking=True)
        options.update(tokenize=False, add_generation_prompt=generation)
        try:
            rendered = self.tokenizer.apply_chat_template(values, **options)
        except TemplateError as error:
            raise validation_error("messages", str(error)) from None
        return self._encode(rendered)

    async def _prepare_image(self, image, tokens):
        if self.images is None:
            raise validation_error("image", "The configured model does not support images")
        return await self.images.prepare(image, tokens)

    def _context(self, needed, field):
        if needed > self.context_window:
            raise validation_error(field,
                f"Input and output need {needed} tokens; the model's context window is {self.context_window}")

    async def _read(self, iterator, tokens, features, labels):
        result = None
        async for output in iterator:
            if not output.finished:
                continue
            if features:
                value = output.outputs.data
                width = self.head.metadata["hidden_size"]
                if not isinstance(value, torch.Tensor) or value.shape != (width + len(labels or ()),):
                    raise RuntimeError("Engine feature shape does not match the decision head")
                value = value.detach()
            else:
                choice, = output.outputs
                value = {"tokens": tuple(choice.token_ids), "text": choice.text,
                         "finish_reason": choice.finish_reason}
            result = value, {"submitted_tokens": len(tokens),
                             "num_cached_tokens": getattr(output, "num_cached_tokens", None)}
        if result is None:
            raise RuntimeError("Engine stream ended without a final response")
        return result

    async def _cleanup(self, request_id, iterator, failed):
        try:
            if failed:
                await self.engine.abort(request_id)
        finally:
            if iterator is not None:
                await iterator.aclose()

    async def _engine_call(self, tokens, params, *, features, multimodal=None, baseline_token_ids=None):
        self._check()
        request_id = "fuzev_" + uuid4().hex
        iterator = None
        failed = True
        try:
            if multimodal is None:
                prompt = {"type": "token", "prompt_token_ids": list(tokens)}
            else:
                self.images.check_count(multimodal)
                prompt = multimodal.engine_input(tokens)
            method = self.engine.encode if features else self.engine.generate
            iterator = method(prompt, params, request_id)
            result = await self._read(iterator, tokens, features, baseline_token_ids)
            failed = False
            return result
        except Exception as error:
            self._raise_if_engine_dead(error)
            raise
        finally:
            await self._cleanup(request_id, iterator, failed)

    def _raise_if_engine_dead(self, error):
        if getattr(self.engine, "errored", False):
            raise APIError(503, "Engine is unavailable") from error

    async def _generation_input(self, request, image):
        template_kwargs = {**self.config.generation_chat_template_kwargs, **request.chat_template_kwargs}
        if request.prompt is not None:
            if image is not None and self.images is None:
                raise validation_error("image", "The configured model does not support images")
            prompt = self._encode((self.images.placeholder if image is not None else "") + request.prompt)
        else:
            prompt = self._messages([item.model_dump() for item in request.messages], generation=True,
                image=image is not None, chat_template_kwargs=template_kwargs)
        multimodal = None
        if image is not None:
            prompt, multimodal = await self._prepare_image(image, prompt)
        available = self.context_window - len(prompt)
        self._context(len(prompt) + (request.max_tokens or 1), "max_tokens" if request.max_tokens is not None
                      else "prompt" if request.prompt is not None else "messages")
        return prompt, multimodal, template_kwargs, available

    async def generate(self, request: GenerationRequest, request_id, *, image=None):
        async with self._operation(request_id):
            prompt, multimodal, template_kwargs, available = await self._generation_input(request, image)
            result, stats = await self._engine_call(prompt, self.api.sampling(request, self.sampling_defaults,
                available, self.reasoning_parser_class), features=False, multimodal=multimodal)
            completion = result["tokens"]
            choice = {"index": 0, "finish_reason": result["finish_reason"]}
            if request.messages is not None:
                reasoning, content = None, result["text"]
                if self.reasoning_parser_class is not None:
                    reasoning, content = self.api.reasoning(self.reasoning_parser_class, self.tokenizer,
                        content, request, template_kwargs, prompt)
                choice["message"] = {"role": "assistant", "content": content, "reasoning": reasoning}
            else:
                choice["text"] = result["text"]
            response = {"id": request_id, "model": self.config.model_name, "created": int(time.time()),
                "object": "chat.completion" if request.messages is not None else "text_completion",
                "choices": [choice], "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                    "total_tokens": len(prompt) + len(completion)}}
            if stats["num_cached_tokens"] is not None:
                response["usage"]["prompt_tokens_details"] = {"cached_tokens": stats["num_cached_tokens"]}
            if request.retain_decision_prefix:
                accepted = prompt + completion
                response["prefix_id"] = self.prefixes.retain(accepted, payload=multimodal).id
            return response

    async def generate_stream(self, request, request_id, *, image=None):
        async with stream_in_task(self._generate_stream(request, request_id, image=image)) as (_, chunks):
            async for chunk in chunks:
                yield chunk

    async def _generate_stream(self, request, request_id, *, image=None):
        async with self._operation(request_id):
            prompt, multimodal, template_kwargs, available = await self._generation_input(request, image)
            chat = request.messages is not None
            parser = (self.api.reasoning_parser(self.reasoning_parser_class, self.tokenizer,
                      template_kwargs, prompt) if chat and self.reasoning_parser_class is not None else None)
            deltas_only = getattr(parser, "engine_based_streaming", False)
            engine_input = ({"type": "token", "prompt_token_ids": list(prompt)} if multimodal is None
                            else multimodal.engine_input(prompt))
            if multimodal is not None:
                self.images.check_count(multimodal)
            engine_id = "fuzev_" + uuid4().hex
            iterator = self.engine.generate(engine_input, self.api.sampling(request, self.sampling_defaults,
                available, self.reasoning_parser_class), engine_id)
            base = {"id": request_id, "model": self.config.model_name, "created": int(time.time()),
                    "object": "chat.completion.chunk" if chat else "text_completion"}
            text, completion, finished, role_sent = "", [], False, False
            try:
                async for output in iterator:
                    choice, = output.outputs
                    delta_text, delta_tokens = choice.text, list(choice.token_ids)
                    if chat:
                        if parser is not None:
                            # Engine-based parsers keep their own state and read only the deltas.
                            previous_text, previous_tokens = ("", []) if deltas_only else (text, completion)
                            parsed = parser.extract_reasoning_streaming(previous_text, previous_text + delta_text,
                                delta_text, previous_tokens, previous_tokens + delta_tokens, delta_tokens)
                            delta = {} if parsed is None else parsed.model_dump(exclude_none=True)
                        else:
                            delta = {"content": delta_text}
                        if delta and not role_sent:
                            delta["role"], role_sent = "assistant", True
                        body = {"delta": delta}
                    else:
                        body = {"text": delta_text}
                    if parser is not None and not deltas_only:
                        text += delta_text
                    completion.extend(delta_tokens)
                    if delta_text or (chat and body["delta"]):
                        yield {**base, "choices": [{"index": 0, **body, "finish_reason": None}]}
                    if output.finished:
                        if parser is not None and hasattr(parser, "finish_streaming"):
                            trailing = parser.finish_streaming()
                            if trailing is not None:
                                yield {**base, "choices": [{"index": 0,
                                    "delta": trailing.model_dump(exclude_none=True), "finish_reason": None}]}
                        finished = True
                        final = {**base, "choices": [{"index": 0,
                            **({"delta": {}} if chat else {"text": ""}), "finish_reason": choice.finish_reason}],
                            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                                      "total_tokens": len(prompt) + len(completion)}}
                        if request.retain_decision_prefix:
                            final["prefix_id"] = self.prefixes.retain(prompt + tuple(completion), payload=multimodal).id
                        yield final
                if not finished:
                    raise RuntimeError("Engine stream ended without a final response")
            except Exception as error:
                self._raise_if_engine_dead(error)
                raise
            finally:
                await drain_tasks([asyncio.create_task(self._cleanup(engine_id, iterator, not finished))])

    async def _decision_input(self, request, image):
        if request.prefix_id is not None:
            prefix = self._get_prefix(request.prefix_id)
            accepted, multimodal = prefix.token_ids, prefix.payload
            if image is not None:
                # The fixed encoding ends this shared prefix with <|vision_end|>.
                candidates = [item.model_dump() for item in request.candidates]
                header, marker, _ = query_suffix(request.question, candidates,
                                                 IMAGE_PLACEHOLDER).partition(IMAGE_PLACEHOLDER)
                tokens, payload = await self._prepare_image(image, self._encode(header + marker))
                multimodal = (multimodal or MultimodalPayload(())).append(payload, len(accepted))
                accepted += tokens
            return accepted, multimodal
        tokens = self._messages(request.messages, generation=False, image=image is not None)
        return await self._prepare_image(image, tokens) if image is not None else (tokens, None)

    async def prepare_decisions(self, requests, *, image=None):
        prepared = await self._decision_input(requests[0], image)

        async def submit(request, request_id):
            return await self.decide(request, request_id, image=image if request.prefix_id is not None else None,
                                     prepared=prepared)
        return submit

    async def decide(self, request, request_id, *, image=None, prepared=None):
        async with self._operation(request_id):
            candidates = [item.model_dump() for item in request.candidates]
            accepted, multimodal = prepared if prepared is not None else await self._decision_input(request, image)
            additional = request.additional_context
            if request.prefix_id is not None and image is not None:
                _, _, tail = query_suffix(request.question, candidates,
                                          IMAGE_PLACEHOLDER + "\n" + additional).partition(IMAGE_PLACEHOLDER)
                suffix = self._encode(tail)
            else:
                suffix = self._encode(query_suffix(request.question, candidates, additional))
            tails = [self._encode(candidate_suffix(candidate)) for candidate in candidates]
            query = accepted + suffix
            decision_overflow(len(query) + max(map(len, tails)), self.context_window)
            option_labels = self.label_ids[:len(candidates)]
            query_value, query_stats = await self._engine_call(query, self.api.pooling(option_labels),
                features=True, multimodal=multimodal, baseline_token_ids=option_labels)
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self._engine_call(query + tail, self.api.pooling(),
                    features=True, multimodal=multimodal)) for tail in tails]
            values = [task.result() for task in tasks]
            width = self.head.metadata["hidden_size"]
            logits, raw, calibrated = self.head.score(request.primitive, query_value[:width],
                torch.stack([feature for feature, _ in values]), baseline=query_value[width:])
            return {"id": request_id, "model": self.config.model_name, "primitive": request.primitive,
                "head_id": self.head.head_id, "calibration_id": self.head.calibration_id,
                "selected_id": candidates[max(range(len(logits)), key=logits.__getitem__)]["id"],
                "scores": [{"id": candidate["id"], "logit": logit, "raw_probability": p,
                    "calibrated_probability": q} for candidate, logit, p, q in zip(candidates, logits, raw, calibrated)],
                "execution": {"submitted_tokens": query_stats["submitted_tokens"] +
                    sum(stats["submitted_tokens"] for _, stats in values)}}

    async def close(self):
        if self.closed:
            return
        self.closed = True
        tasks = list(self.operations.values())
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        try:
            await drain_tasks(tasks)
        finally:
            self.prefixes.clear()
            if self.head is not None:
                self.head.close()
        # The injected engine remains the caller's responsibility.
