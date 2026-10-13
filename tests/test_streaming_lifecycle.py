"""CPU regressions for stream ownership and disconnected clients."""

import asyncio
import json
import threading

import httpx
import pytest

from fuzev.images import decode_image
from fixtures import application, generation, image_string, service, until


def request_stream(holder, send, *, spec="2.0", image=None):
    incoming = asyncio.Queue()
    body = {"prompt": "Question", "stream": True, "max_tokens": 8}
    if image is not None:
        body["image"] = image
    incoming.put_nowait({"type": "http.request", "body": json.dumps(body).encode(), "more_body": False})
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec},
             "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/v1/completions", "raw_path": b"/v1/completions", "root_path": "",
             "query_string": b"", "headers": [(b"x-typesafe-request-id", b"lifecycle")],
             "client": ("127.0.0.1", 1), "server": ("fixture", 80)}
    return asyncio.create_task(holder.app(scope, incoming.get, send)), incoming


def test_disconnect_before_first_chunk_aborts_without_waiting_for_generation():
    async def case():
        holder = application()
        async with holder.app.router.lifespan_context(holder.app):
            holder.engine.wait = asyncio.Event()
            output = []

            async def send(message):
                output.append(message)

            pending, incoming = request_stream(holder, send)
            try:
                await holder.engine.entered.wait()
                incoming.put_nowait({"type": "http.disconnect"})
                await asyncio.wait({pending}, timeout=.5)
                assert pending.done(), "Disconnect must interrupt the first-chunk wait"
                await pending
                assert output[0]["status"] == 499
                assert holder.engine.aborted and not holder.engine.active
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(asyncio.wait_for(case(), 5))


@pytest.mark.parametrize("spec", ["2.0", "2.4"])
@pytest.mark.parametrize("failed_message", ["http.response.start", "http.response.body"])
def test_send_failure_closes_prefetched_stream(spec, failed_message):
    async def case():
        holder = application()
        async with holder.app.router.lifespan_context(holder.app):
            holder.engine.stream_wait = asyncio.Event()

            async def send(message):
                if message["type"] == failed_message:
                    raise OSError("Client socket closed")

            pending, _ = request_stream(holder, send, spec=spec)
            await asyncio.gather(pending, return_exceptions=True)
            assert holder.engine.aborted and not holder.engine.active
    asyncio.run(asyncio.wait_for(case(), 5))


@pytest.mark.parametrize("separate_consumer", [False, True])
def test_service_close_owns_stream_after_first_chunk(separate_consumer):
    async def case():
        svc = service()
        svc.engine.stream_wait = asyncio.Event()
        iterator = svc.generate_stream(generation(stream=True), "parked")
        try:
            first = await (asyncio.create_task(anext(iterator)) if separate_consumer else anext(iterator))
            assert first["choices"][0]["text"] == "fixture reasoning"
            await svc.close()
            assert svc.engine.aborted and not svc.engine.active
            assert svc.head.closed
            assert not svc.engine.shutdown_called
        finally:
            await iterator.aclose()
    asyncio.run(asyncio.wait_for(case(), 5))


@pytest.mark.parametrize("phase", ["before-first", "after-first", "blocked-send"])
def test_app_close_drains_stream_before_head_close(phase):
    async def case():
        holder = application()
        lifetime = holder.app.router.lifespan_context(holder.app)
        await lifetime.__aenter__()
        holder.engine.abort_wait = asyncio.Event()
        holder.engine.stream_wait = asyncio.Event()
        if phase == "before-first":
            holder.engine.wait = asyncio.Event()
        streamed = asyncio.Event()

        async def send(message):
            if b"fixture reasoning" in message.get("body", b""):
                streamed.set()
                if phase == "blocked-send":
                    await asyncio.Event().wait()

        pending, _ = request_stream(holder, send)
        closing = None
        try:
            await (holder.engine.entered.wait() if phase == "before-first" else streamed.wait())
            closing = asyncio.create_task(lifetime.__aexit__(None, None, None))
            await holder.engine.aborting.wait()
            assert not holder.service.head.closed and not closing.done()
            holder.engine.abort_wait.set()
            await closing
            if phase == "blocked-send":
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            assert holder.service.head.closed and holder.context_exited
            assert not holder.engine.active and not holder.engine.shutdown_called
        finally:
            holder.engine.abort_wait.set()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            if closing is None:
                await lifetime.__aexit__(None, None, None)
            else:
                await closing
    asyncio.run(asyncio.wait_for(case(), 5))


def test_cancel_during_image_decode_closes_the_returned_image(monkeypatch):
    import fuzev_vllm.server as server

    entered, release = threading.Event(), threading.Event()
    images = []

    def decode(value):
        image = decode_image(value)
        images.append(image)
        entered.set()
        assert release.wait(3)
        return image

    monkeypatch.setattr(server, "decode_image", decode)

    async def case():
        holder = application()
        async with holder.app.router.lifespan_context(holder.app):
            async def send(message):
                pass

            pending, _ = request_stream(holder, send, image=image_string())
            try:
                await until(entered.is_set)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=holder.app),
                                             base_url="http://fixture") as client:
                    assert (await client.post("/v1/requests/lifecycle/cancel")).status_code == 202
                await asyncio.sleep(.02)
                assert not pending.done(), "The decoder still owns an image"
                release.set()
                await pending
                with pytest.raises(ValueError, match="closed"):
                    images[0].getpixel((0, 0))
                assert not holder.engine.calls
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)
    asyncio.run(asyncio.wait_for(case(), 5))
