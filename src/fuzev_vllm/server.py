"""Serve generation and FuZev decisions with vLLM."""

import argparse
import asyncio
import json
from contextlib import aclosing, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse

from fuzev.api import (APIBoundary, APIError, add_decision_route, finished_before_disconnect, health_response,
                       models_response, parse_body, sse_event, stream_error, validation_error)
from fuzev.images import decode_image
from fuzev.prefix import DEFAULT_PREFIX_CAPACITY
from fuzev.service_contracts import GenerationRequest
from fuzev.systemone import SystemOneAdapter, SystemOneRequest

from . import __version__
from .service import VllmServiceConfig, build_service, drain_tasks, stream_in_task


class _StreamingResponse(StreamingResponse):
    def __init__(self, content, close_stream):
        super().__init__(content, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        self.close_stream = close_stream

    async def __call__(self, scope, receive, send):
        async def cleanup():
            try:
                await self.body_iterator.aclose()
            finally:
                await self.close_stream()

        try:
            await super().__call__(scope, receive, send)
        finally:
            await drain_tasks([asyncio.create_task(cleanup())])


def create_app(engine_context, config, *, builder=build_service, api_key=None, max_request_bytes=None):
    @asynccontextmanager
    async def lifespan(app):
        async with engine_context() as engine:
            service = await builder(engine, config)
            app.state.service = service
            app.state.tasks = {}
            app.state.closing = False
            if service.head is not None:
                app.state.systemone = SystemOneAdapter(
                    service.decide, service.config.model_name,
                    lambda text: len(service.tokenizer.encode(text, add_special_tokens=False, truncation=False)),
                    head_id=service.head.head_id, max_options=service.head.max_options,
                    primitives=service.head.primitives)

            try:
                yield
            finally:
                app.state.closing = True
                pending = list(app.state.tasks.values())
                for task in pending:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                try:
                    await drain_tasks(pending)
                finally:
                    await service.close()

    app = FastAPI(title="FuZev vLLM server", version=__version__, lifespan=lifespan)
    app.add_middleware(APIBoundary, api_key=api_key,
                       **({} if max_request_bytes is None else {"max_request_bytes": max_request_bytes}))

    async def execute(http, function):
        if app.state.closing:
            raise APIError(503, "Server is closing")
        key = http.state.request_id
        tasks = app.state.tasks
        if key in tasks:
            raise APIError(409, "Request ID is already active")
        task = asyncio.create_task(function(key))
        tasks[key] = task
        try:
            if not await finished_before_disconnect(http, task):
                if not task.cancelling():
                    task.cancel()
                await drain_tasks([task])
                raise APIError(499, "Client disconnected")
            return task.result()
        except asyncio.CancelledError:
            if not task.done() and not task.cancelling():
                task.cancel()
            await drain_tasks([task])
            if asyncio.current_task().cancelling():
                raise
            raise APIError(499, "Request cancelled") from None
        finally:
            tasks.pop(key, None)

    @asynccontextmanager
    async def request_image(request):
        if request.image is None:
            yield None
            return
        task = asyncio.create_task(asyncio.to_thread(decode_image, request.image))
        try:
            await drain_tasks([task])
            yield task.result()
        finally:
            if task.done() and not task.cancelled() and task.exception() is None:
                task.result().close()

    async def with_image(request, function):
        async with request_image(request) as image:
            return await function(image)

    @app.post("/v1/completions")
    @app.post("/v1/chat/completions")
    async def generate(http: Request):
        request = await parse_body(http, GenerationRequest)
        if (http.url.path == "/v1/chat/completions") != (request.messages is not None):
            raise validation_error("messages", "Chat requires messages; completions require prompt")
        service = app.state.service
        if request.stream:
            return await stream(http, request)
        return await execute(http, lambda key: with_image(request,
            lambda image: service.generate(request, key, image=image)))

    async def stream(http, request):
        key, tasks = http.state.request_id, app.state.tasks
        if app.state.closing:
            raise APIError(503, "Server is closing")
        if key in tasks:
            raise APIError(409, "Request ID is already active")

        async def events():
            async with request_image(request) as image:
                async with aclosing(app.state.service.generate_stream(request, key, image=image)) as iterator:
                    async for chunk in iterator:
                        yield sse_event(chunk)

        context = stream_in_task(events())
        task, iterator = await context.__aenter__()
        tasks[key] = task
        prefetch = asyncio.create_task(anext(iterator))

        async def cleanup():
            if not prefetch.done():
                prefetch.cancel()
            try:
                await drain_tasks([prefetch])
                await context.__aexit__(None, None, None)
            finally:
                tasks.pop(key, None)

        try:
            if not await finished_before_disconnect(http, prefetch):
                raise APIError(499, "Client disconnected")
            first = prefetch.result()
        except BaseException as error:
            await drain_tasks([asyncio.create_task(cleanup())])
            if isinstance(error, asyncio.CancelledError) and not asyncio.current_task().cancelling():
                raise APIError(499, "Request cancelled") from None
            raise

        async def chunks():
            try:
                yield first
                async for chunk in iterator:
                    yield chunk
            except (Exception, asyncio.CancelledError) as error:
                if isinstance(error, asyncio.CancelledError) and (
                        not task.cancelled() or asyncio.current_task().cancelling()):
                    raise
                yield stream_error(APIError(499, "Request cancelled") if isinstance(
                    error, asyncio.CancelledError) else error)
            yield sse_event("[DONE]")

        return _StreamingResponse(chunks(), cleanup)

    async def respond(http, operation):
        return await execute(http, lambda key: operation(app.state.service, key))

    async def submit_decision(service, request, key):
        return await with_image(request, lambda image: service.decide(request, key, image=image))

    add_decision_route(app, respond, submit_decision)

    if config.head_directory:
        @app.post("/v1/systemone")
        async def systemone(http: Request):
            request = await parse_body(http, SystemOneRequest)
            service = app.state.service

            async def answer(key):
                async def decoded(image):
                    return await app.state.systemone.answer(request,
                        prepare=lambda prepared: service.prepare_decisions(prepared, image=image))
                return await with_image(request, decoded)

            return await execute(http, answer)

    @app.delete("/v1/prefixes/{prefix_id}")
    async def release(prefix_id: str):
        return {"id": prefix_id, "released": app.state.service.release_prefix(prefix_id)}

    @app.post("/v1/requests/{key}/cancel", status_code=202)
    async def cancel(key: str):
        task = app.state.tasks.get(key)
        if task is None or task.done():
            raise APIError(404, "Unknown or completed request")
        if not task.cancelling():
            task.cancel()
        return {"id": key, "cancel_requested": True}

    @app.get("/v1/models")
    async def models():
        service = app.state.service
        return models_response(service.config.model_name, service.head)

    @app.get("/health")
    async def health():
        service = app.state.service
        service._check()
        try:
            await service.engine.check_health()
        except Exception as error:
            raise APIError(503, "Engine is unavailable") from error
        return health_response(service.config.model_name, service.context_window,
                               decisions=service.head is not None, images=service.images is not None)

    return app


def engine_context(engine_options, *, decisions=True):
    options = dict(engine_options)
    additional = dict(options.get("additional_config") or {})
    additional["fuzev"] = {**additional.get("fuzev", {}), "enabled": decisions}
    options["additional_config"] = additional

    @asynccontextmanager
    async def context():
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.usage.usage_lib import UsageContext
        engine = AsyncLLM.from_vllm_config(
            AsyncEngineArgs(**options).create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER),
            usage_context=UsageContext.OPENAI_API_SERVER)
        try:
            yield engine
        finally:
            engine.shutdown()

    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-config", required=True, help="JSON file containing AsyncEngineArgs")
    parser.add_argument("--model-name", required=True, help="Model ID advertised by this server")
    parser.add_argument("--head-directory", help="Decision head bundle; omit for generation only")
    parser.add_argument("--head-device", help="PyTorch device for decision scoring (default cpu)")
    parser.add_argument("--calibration", choices=("raw", "calibrated"), help="Score calibration (default raw)")
    parser.add_argument("--reasoning-parser", help=(
        "vLLM reasoning parser name, or none (default qwen3, for Qwen3-style thinking models; "
        "other models need their own parser or none, otherwise output without </think> returns content null)"))
    parser.add_argument("--chat-template-kwargs", default="{}", help="Generation template arguments as JSON")
    parser.add_argument("--prefix-capacity", type=int,
                        help=f"Retained prefixes with LRU eviction (default {DEFAULT_PREFIX_CAPACITY})")
    parser.add_argument("--max-request-bytes", type=int, help="Maximum HTTP request body size in bytes (default 8 MiB)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", help="Optional bearer token required on /v1 routes")
    args = parser.parse_args()
    supplied = {name: value for name, value in (("head_device", args.head_device), ("calibration", args.calibration),
                ("prefix_capacity", args.prefix_capacity)) if value is not None}
    if args.reasoning_parser is not None:
        supplied["reasoning_parser"] = None if args.reasoning_parser == "none" else args.reasoning_parser
    config = VllmServiceConfig(model_name=args.model_name, head_directory=args.head_directory,
        generation_chat_template_kwargs=json.loads(args.chat_template_kwargs), **supplied)
    options = json.loads(Path(args.engine_config).read_text())
    app = create_app(engine_context(options, decisions=bool(config.head_directory)), config,
                     api_key=args.api_key, max_request_bytes=args.max_request_bytes)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
