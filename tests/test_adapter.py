"""CPU behavior tests for the public API and shared-engine adapter."""

import asyncio
import json
import os
import subprocess
import sys
from collections import Counter
from types import SimpleNamespace as NS

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from fuzev.api import APIError
from fuzev.encoding import candidate_suffix
from fuzev.systemone import SystemOneRequest, decision_request
from fuzev_vllm.service import VllmAPI, build_service
from fixtures import (API, Engine, FixtureHead, application, config, decision,
                      generation, image_string, service, until)


DECISION_KEYS = {"id", "model", "primitive", "head_id", "calibration_id", "selected_id", "scores", "execution"}
SYSTEMONE = {"state": {"ticket": "Charged twice", "urgent": True}, "model": "jev-preview-2026",
    "questions": {
        "team": {"type": "choice", "criteria": {"billing": "Charges", "shipping": None}},
        "truth": {"type": "noul", "criteria": {"true": "Urgent", "false": "Not urgent"}},
        "score": {"type": "score", "criteria": ["low", "middle", {"level": "high"}]}}}


def test_real_sdk_round_trip():
    import httpx2
    import typesafe_sdk as sdk

    async def case():
        holder = application()
        async with holder.app.router.lifespan_context(holder.app):
            async with sdk.AsyncTypeSafeClient(api_key="test", base_url="http://fixture",
                    transport=httpx2.ASGITransport(app=holder.app),
                    retry=sdk.RetryPolicy(max_retries=0)) as client:
                models = await client.models.list()
                assert models.models[0].name == "fixture-only"
                result = await client.system_one(state=SYSTEMONE["state"],
                    questions={"choice": sdk.Choice(criteria={"billing": "Charges", "shipping": "Parcels"}),
                               "noul": sdk.Noul(instructions="Was the customer charged twice?"),
                               "score": sdk.Score(criteria=["low", "high"])},
                    model="jev-latest", extra_headers={"X-TypeSafe-Request-ID": "sdk.vllm"})
                assert result.model == "fixture-only" and result.request_id == "sdk.vllm"
                assert set(result.choices["choice"].probabilities) == {"billing", "shipping"}
                assert 0 <= result.nouls["noul"].noul <= 1
                assert result.scores["score"].legend == {0: "low", 1: "high"}
    asyncio.run(case())


def test_shared_engine_compact_decisions():
    async def case():
        svc = service()
        candidates = [{"id": "a", "text": "low"}, {"id": "b", "text": "high"}]
        request = decision(primitive="choice", candidates=candidates)
        result = await svc.decide(request, "test.1:decision")
        assert set(result) == DECISION_KEYS
        assert result["calibration_id"] == "raw"
        assert result["execution"] == {"submitted_tokens": sum(len(row[1]) for row in svc.engine.calls)}
        query, *branches = svc.engine.calls
        assert [branch[1] for branch in branches] == [
            query[1] + tuple(svc.tokenizer.encode(candidate_suffix(candidate), add_special_tokens=False))
            for candidate in candidates]
        assert [score["id"] for score in result["scores"]] == [candidate["id"] for candidate in candidates]
        assert [score["logit"] for score in result["scores"]] == [50.0, 50.25]
        expected = torch.softmax(torch.tensor([50.0, 50.25]), 0).tolist()
        for field in ("raw_probability", "calibrated_probability"):
            assert [score[field] for score in result["scores"]] == pytest.approx(expected)
        await svc.close()
        assert svc.head.closed and not svc.engine.shutdown_called
    asyncio.run(case())


def test_generation_unused_head_and_reasoning_and_exact_prefix():
    async def case():
        svc = service()
        response = await svc.generate(generation(prompt=None,
            messages=[{"role": "user", "content": "Answer"}], chat_template_kwargs={"enable_thinking": False}), "chat")
        assert set(response) == {"id", "model", "created", "object", "choices", "usage"}
        assert response["choices"][0]["message"] == {
            "role": "assistant", "content": "fixture answer", "reasoning": "fixture reasoning"}
        assert svc.api.reasoning_args[4] == {"enable_thinking": False}
        assert not svc.head.observed
        assert all(row[0] == "generate" for row in svc.engine.calls)
        generated = await svc.generate(generation(retain_decision_prefix=True), "retain")
        accepted = svc.engine.calls[-1][1] + tuple(svc.engine.completion)
        prefix = svc.prefixes.get(generated["prefix_id"])
        assert prefix.token_ids == accepted
        await svc.decide(decision(messages=None, prefix_id=prefix.id), "follow")
        assert svc.release_prefix(prefix.id)
        with pytest.raises(APIError) as error:
            svc.release_prefix(prefix.id)
        assert error.value.status == 404
        await svc.close()
    asyncio.run(case())


def test_sampling_inherits_model_defaults_and_optional_parameters():
    from vllm.sampling_params import SamplingParams
    defaults = {"temperature": .3, "top_p": .7, "repetition_penalty": 1.2}
    request = generation(max_tokens=None, temperature=None, top_p=None)
    params = VllmAPI.sampling(request, defaults, 12345)
    assert isinstance(params, SamplingParams)
    assert (params.temperature, params.top_p, params.repetition_penalty, params.max_tokens) == (.3, .7, 1.2, 12345)
    params = VllmAPI.sampling(generation(temperature=0., seed=4, max_tokens=7), defaults, 12345)
    assert params.temperature == 0. and params.seed == 4 and params.max_tokens == 7


@pytest.mark.parametrize("text,kwargs,expected", [
    ("<think>consider</think>answer", {}, ("consider", "answer")),
    ("consider</think>answer", {}, ("consider", "answer")),
    ("answer", {"enable_thinking": False}, (None, "answer")),
    ("<think>  consider \n</think>  answer \n", {}, ("  consider \n", "  answer \n")),
    ("still thinking", {}, ("still thinking", None)),
])
def test_actual_upstream_qwen3_reasoning_parser(text, kwargs, expected):
    tokenizer = NS(get_vocab=lambda: {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4})
    request = generation(prompt=None, messages=[{"role": "user", "content": "Question"}],
                         chat_template_kwargs=kwargs)
    parser_class = VllmAPI.reasoning_parser_class("qwen3")
    assert VllmAPI.reasoning(parser_class, tokenizer, text, request, kwargs, ()) == expected


def test_systemone_matches_decisions_api():
    holder = application()
    with TestClient(holder.app) as client:
        payload = {**SYSTEMONE, "future_field": True,
                   "questions": {key: {**q, "future_question_field": 3} for key, q in SYSTEMONE["questions"].items()}}
        result = client.post("/v1/systemone", json=payload, headers={"X-TypeSafe-Request-ID": "sdk.1:_-"})
        assert result.status_code == 200, result.text
        assert result.headers["x-typesafe-request-id"] == "sdk.1:_-"
        body = result.json()
        assert body["model"] == "fixture-only"
        assert list(body["answers"]) == list(SYSTEMONE["questions"])
        request = SystemOneRequest.model_validate(payload)
        original_calls = Counter((row[0], row[1]) for row in holder.engine.calls)
        offset = len(holder.engine.calls)
        total = 0
        for key, question in request.questions.items():
            from fuzev.systemone import state_text
            direct = decision_request(request, question, "fixture-only", "reference", state_text(request.state))
            response = client.post("/v1/decisions", json=direct.model_dump(mode="json"))
            assert response.status_code == 200, response.text
            direct_body = response.json()
            assert set(direct_body) == DECISION_KEYS
            total += direct_body["execution"]["submitted_tokens"]
            probs = [row["calibrated_probability"] for row in direct_body["scores"]]
            answer = body["answers"][key]
            if question.type == "noul":
                assert [item.id for item in direct.candidates] == ["yes", "no"]
                assert [item.text for item in direct.candidates] == ["Yes: Urgent", "No: Not urgent"]
                assert direct.question == ""
                assert answer["noul"] == probs[0]
            elif question.type == "choice":
                assert answer["probabilities"] == dict(zip(question.criteria, probs))
                assert answer["choice"] == direct_body["selected_id"]
            else:
                assert answer["probabilities"] == {str(i): p for i, p in enumerate(probs)}
                assert answer["score"] == pytest.approx(sum(i * p for i, p in enumerate(probs)))
        assert body["usage"]["input_tokens"] == total
        assert original_calls == Counter((row[0], row[1]) for row in holder.engine.calls[offset:])
        rendered = holder.engine.tokenizer.renders[0][0][0]["content"]
        assert rendered == "ticket: Charged twice\nurgent: True"
        models = client.get("/v1/models", headers={"X-TypeSafe-Request-ID": "models:1"})
        assert models.headers["x-typesafe-request-id"] == "models:1"
        assert models.json()["models"][0]["release_date"] == "2026-10-12"
        health = client.get("/health")
        assert set(health.json()) == {"status", "model", "context_window", "capabilities"}
        assert set(health.json()["capabilities"]) == {"generation", "decisions", "images", "streaming"}
    assert holder.service.head.closed and not holder.engine.shutdown_called
    assert holder.context_exited


def test_many_questions_are_concurrent_and_keep_order():
    async def case():
        holder = application()
        async with holder.app.router.lifespan_context(holder.app):
            holder.engine.wait = asyncio.Event()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=holder.app), base_url="http://test") as client:
                n = 20
                questions = {f"q{i}": {"type": "noul"} for i in range(n)}
                task = asyncio.create_task(client.post("/v1/systemone", json={"state": "x", "questions": questions}))
                await until(lambda: holder.engine.active == n)
                holder.engine.wait.set()
                result = await task
                assert result.status_code == 200, result.text
                assert list(result.json()["answers"]) == list(questions)
            assert not holder.engine.active
    asyncio.run(case())


def test_public_cancel_drains_all_children_and_duplicate_id_conflicts():
    async def case():
        holder = application()
        async with holder.app.router.lifespan_context(holder.app):
            engine = holder.engine
            engine.wait, engine.abort_wait = asyncio.Event(), asyncio.Event()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=holder.app), base_url="http://test") as client:
                headers = {"X-TypeSafe-Request-ID": "public.1:cancel"}
                task = asyncio.create_task(client.post("/v1/systemone", json=SYSTEMONE, headers=headers))
                await until(lambda: engine.active == 3)
                duplicate = await client.post("/v1/completions", json=generation().model_dump(), headers=headers)
                assert duplicate.status_code == 409
                first = await client.post("/v1/requests/public.1:cancel/cancel")
                assert first.status_code == 202 and set(first.json()) == {"id", "cancel_requested"}
                await engine.aborting.wait()
                second = await client.post("/v1/requests/public.1:cancel/cancel")
                assert second.status_code == 202 and not task.done()
                engine.abort_wait.set()
                result = await task
                assert result.status_code == 499 and result.headers["x-typesafe-request-id"] == "public.1:cancel"
                assert (await client.post("/v1/requests/public.1:cancel/cancel")).status_code == 404
                engine.wait.set()
                assert (await client.post("/v1/completions", json=generation().model_dump())).status_code == 200
    asyncio.run(case())


@pytest.mark.parametrize("path,payload,status", [
    ("/v1/completions", {"messages": [{"role": "user", "content": "Hi"}]}, 422),
    ("/v1/chat/completions", {"prompt": "Hi"}, 422),
    ("/v1/decisions", decision(messages=None, prefix_id="missing").model_dump(), 404),
    ("/v1/completions", {"model": "fixture-only", "prompt": "x", "max_tokens": Engine().vllm_config.model_config.max_model_len}, 422),
    ("/v1/systemone", {"state": "x", "questions": {"one": {"type": "choice", "criteria": {"a": None}}}}, 422),
    ("/v1/systemone", {"state": "x", "questions": {"one": {"type": "score", "criteria": ["a"]}}}, 422),
    ("/v1/systemone", {"state": "x", "questions": {"one": {"type": "score", "criteria": "wrong"}}}, 422),
    ("/v1/decisions", {**decision().model_dump(), "candidates": [{"id": "one", "text": "Only"}]}, 422),
    ("/v1/completions", {"model": "fixture-only", "prompt": "x", "image": "not_base64"}, 422),
])
def test_public_errors(path, payload, status):
    holder = application()
    with TestClient(holder.app) as client:
        response = client.post(path, json=payload, headers={"X-TypeSafe-Request-ID": "error:1"})
        assert response.status_code == status, response.text
        assert response.headers["x-typesafe-request-id"] == "error:1"
        if status == 422:
            assert isinstance(response.json()["detail"], list)
        assert not holder.engine.calls


def test_internal_errors_are_not_misreported_as_client_errors():
    holder = application()
    with TestClient(holder.app, raise_server_exceptions=False) as client:
        holder.engine.failure = ValueError("internal mismatch")
        response = client.post("/v1/completions", json=generation().model_dump())
        assert response.status_code == 500 and response.json() == {"detail": "Internal server error"}


@pytest.mark.parametrize("headers", [
    {"X-TypeSafe-Request-ID": "bad space"},
    [("X-TypeSafe-Request-ID", "one"), ("X-TypeSafe-Request-ID", "two")],
])
@pytest.mark.parametrize("path", ["/health", "/v1/systemone"])
def test_bad_id_errors_are_uniform(path, headers):
    holder = application()
    with TestClient(holder.app) as client:
        response = client.request("POST" if path == "/v1/systemone" else "GET", path, headers=headers, json=SYSTEMONE)
        assert response.status_code == 422
        assert response.headers["x-typesafe-request-id"].startswith("req_")


def test_bad_json_and_finite_json_error():
    holder = application()
    with TestClient(holder.app) as client:
        for body in ('{', '[' * 2000 + '0' + ']' * 2000,
                     '{"state":NaN,"questions":{"q":{"type":"noul"}}}'):
            result = client.post("/v1/systemone", content=body, headers={"X-TypeSafe-Request-ID": "json.1"})
            assert result.status_code == 422 and result.headers["x-typesafe-request-id"] == "json.1"


@pytest.mark.parametrize("maximum,count,status", [(3, 4, 422), (20, 20, 200)])
def test_option_limit_comes_from_loaded_head(maximum, count, status):
    holder = application(head_max_options=maximum)
    with TestClient(holder.app) as client:
        result = client.post("/v1/systemone", json={"state": "x", "questions": {
            "q": {"type": "choice", "criteria": {str(i): None for i in range(count)}}}})
        assert result.status_code == status, result.text
        assert bool(holder.engine.calls) == (status == 200)
        if status == 200:
            assert len(result.json()["answers"]["q"]["probabilities"]) == count
        else:
            assert str(maximum) in result.json()["detail"][0]["msg"]


@pytest.mark.parametrize("path", ["/v1/systemone", "/v1/decisions", "/v1/completions"])
def test_image_inputs_decode_once_at_public_boundary(monkeypatch, path):
    import fuzev_vllm.server as server
    calls = []
    actual = server.decode_image
    def decode(value):
        calls.append(value)
        return actual(value)
    monkeypatch.setattr(server, "decode_image", decode)
    holder = application()
    with TestClient(holder.app) as client:
        payload = (SYSTEMONE if path == "/v1/systemone" else decision().model_dump()
                   if path == "/v1/decisions" else generation().model_dump())
        payload = {**payload, "image": image_string()}
        response = client.post(path, json=payload)
        assert response.status_code == 200, response.text
        assert len(calls) == 1
        assert all(row["type"] == "multimodal" for row in holder.engine.inputs)
        with pytest.raises(ValueError):
            holder.engine.renderer.pil_images[0].getpixel((0, 0))


def test_decision_on_retained_image_prefix():
    holder = application()
    with TestClient(holder.app) as client:
        generated = client.post("/v1/completions", json=generation(
            retain_decision_prefix=True, image=image_string()).model_dump(mode="json"))
        assert generated.status_code == 200, generated.text
        response = client.post("/v1/decisions", json=decision(
            messages=None, prefix_id=generated.json()["prefix_id"]).model_dump(mode="json"))
        assert response.status_code == 200, response.text
        scores = response.json()["scores"]
        assert [score["id"] for score in scores] == ["a", "b"]
        assert [score["calibrated_probability"] for score in scores] == pytest.approx(
            torch.softmax(torch.tensor([50.0, 50.25]), 0).tolist())


def test_headless_server_has_no_systemone_route():
    holder = application(head_directory=None)
    with TestClient(holder.app) as client:
        assert "/v1/systemone" not in client.get("/openapi.json").json()["paths"]
        assert client.post("/v1/systemone", json=SYSTEMONE).status_code == 404
        assert client.get("/v1/models").json()["models"] == []
        assert client.post("/v1/completions", json=generation().model_dump()).status_code == 200
    assert not holder.engine.shutdown_called


@pytest.mark.parametrize("header,status", [(None, 401), ("Bearer wrong", 401), ("Bearer secret", 200)])
def test_api_key_guards_v1_and_health_remains_available(header, status):
    holder = application(api_key="secret")
    headers = {} if header is None else {"Authorization": header}
    with TestClient(holder.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models", headers=headers).status_code == status
        assert client.post("/v1/systemone", json=SYSTEMONE, headers=headers).status_code == status


def test_deeply_nested_json_is_a_client_error():
    holder = application()
    with TestClient(holder.app) as client:
        response = client.post("/v1/systemone", content=b"[" * 100000 + b"]" * 100000)
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "JSON nesting is too deep"


def test_builder_loads_bundle_on_requested_device_and_closes_on_shape_mismatch():
    async def case():
        engine = Engine()
        loaded = []
        def factory(directory, **kwargs):
            head = FixtureHead()
            loaded.append((head, kwargs))
            return head
        svc = await build_service(engine, config(head_device="cpu"), head_factory=factory, api=API())
        assert loaded[0][1]["device"] == "cpu"
        await svc.close()
        engine.vllm_config.model_config.hf_text_config.hidden_size = 16
        with pytest.raises(ValueError, match="feature width"):
            await build_service(engine, config(), head_factory=factory, api=API())
        assert loaded[-1][0].closed
    asyncio.run(case())


def test_shutdown_drains_requests_before_closing_head():
    async def case():
        holder = application()
        lifetime = holder.app.router.lifespan_context(holder.app)
        await lifetime.__aenter__()
        holder.engine.wait, holder.engine.abort_wait = asyncio.Event(), asyncio.Event()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=holder.app), base_url="http://test") as client:
            request = asyncio.create_task(client.post("/v1/systemone", json=SYSTEMONE))
            await until(lambda: holder.engine.active == 3)
            closing = asyncio.create_task(lifetime.__aexit__(None, None, None))
            await holder.engine.aborting.wait()
            assert not holder.service.head.closed
            holder.engine.abort_wait.set()
            assert (await request).status_code == 499
            await closing
            assert holder.service.head.closed and not holder.engine.active
            assert not holder.engine.shutdown_called
    asyncio.run(case())


def test_module_help_runs_without_starting_an_engine():
    result = subprocess.run([sys.executable, "-m", "fuzev_vllm", "--help"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--port" in result.stdout


def test_request_body_limit_and_reserved_template_error():
    holder = application(max_request_bytes=1024)
    with TestClient(holder.app) as client:
        assert client.post("/v1/completions", json={"prompt": "x" * 1024}).status_code == 413
        result = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hi"}],
                             "chat_template_kwargs": {"chat_template": "replacement"}})
        assert result.status_code == 422
        assert result.json()["detail"][0]["loc"] == ["body", "chat_template_kwargs"]
        assert not holder.engine.calls


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_streaming_chunks_preserve_text_reasoning_and_prefix(path):
    holder = application()
    with TestClient(holder.app) as client:
        body = {"messages": [{"role": "user", "content": "Hi"}]} if "chat" in path else {"prompt": "Hi"}
        response = client.post(path, json={**body, "stream": True, "max_tokens": 8,
                                          "retain_decision_prefix": True})
        assert response.status_code == 200, response.text
        events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(value) for value in events[:-1]]
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        if "chat" in path:
            assert "".join(chunk["choices"][0]["delta"].get("reasoning", "") for chunk in chunks) == "fixture reasoning"
            assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == "fixture answer"
        else:
            assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == holder.engine.text
        prefix = chunks[-1]["prefix_id"]
        result = client.post("/v1/decisions", json=decision(messages=None, prefix_id=prefix).model_dump())
        assert result.status_code == 200, result.text


def test_eight_concurrent_http_requests_complete():
    async def case():
        holder = application()
        async with holder.app.router.lifespan_context(holder.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=holder.app), base_url="http://test") as client:
                responses = await asyncio.gather(*[client.post("/v1/systemone", json=SYSTEMONE) for _ in range(8)])
                assert all(response.status_code == 200 for response in responses)
                answers = responses[0].json()["answers"]
                assert all(response.json()["answers"] == answers for response in responses)
                assert not holder.engine.active
    asyncio.run(case())


@pytest.mark.parametrize("disconnect", [True, False], ids=["disconnect", "cancel-route"])
def test_stream_is_incremental_and_cancellation_frees_the_engine(disconnect):
    async def case():
        holder = application()
        async with holder.app.router.lifespan_context(holder.app):
            holder.engine.stream_wait = asyncio.Event()
            queue, output, streamed = asyncio.Queue(), [], asyncio.Event()
            await queue.put({"type": "http.request", "body": json.dumps({"prompt": "Question", "stream": True}).encode(),
                             "more_body": False})
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                     "scheme": "http", "path": "/v1/completions", "raw_path": b"/v1/completions", "root_path": "",
                     "query_string": b"", "headers": [(b"x-typesafe-request-id", b"stream-cancel")],
                     "client": ("127.0.0.1", 1), "server": ("fixture", 80)}
            async def send(message):
                output.append(message)
                if b"fixture reasoning" in message.get("body", b""):
                    streamed.set()
            pending = asyncio.create_task(holder.app(scope, queue.get, send))
            await streamed.wait()
            assert output[0]["status"] == 200 and not pending.done()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=holder.app), base_url="http://test") as client:
                if disconnect:
                    await queue.put({"type": "http.disconnect"})
                else:
                    assert (await client.post("/v1/requests/stream-cancel/cancel")).status_code == 202
                await pending
                assert holder.engine.aborted and not holder.engine.active
                assert (await client.post("/v1/requests/stream-cancel/cancel")).status_code == 404
                assert (await client.post("/v1/completions", json={"prompt": "Next"})).status_code == 200
    asyncio.run(asyncio.wait_for(case(), 5))


def test_unsupported_head_primitive_and_bad_feature_width_are_reported():
    holder = application()
    with TestClient(holder.app) as client:
        holder.service.head.primitives = ("choice",)
        holder.app.state.systemone.primitives = ("choice",)
        response = client.post("/v1/systemone", json=SYSTEMONE)
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "questions", "truth", "type"]
        assert not holder.engine.calls
        def wrong_width(output, kind):
            if kind == "encode":
                output.outputs.data = torch.zeros(3)
        holder.engine.patch_output = wrong_width
        response = client.post("/v1/decisions", json=decision().model_dump())
        assert response.status_code == 500 and holder.engine.aborted


def test_encoding_mismatch_names_both_versions():
    async def case():
        head = FixtureHead()
        head.encoding_version = "other-encoding-v2"
        with pytest.raises(ValueError, match="other-encoding-v2.*fuzev-candidate-branches-v1"):
            await build_service(Engine(), config(), head_factory=lambda *a, **k: head, api=API())
        assert head.closed
    asyncio.run(case())


def test_engine_image_limit_is_a_field_error():
    holder = application()
    with TestClient(holder.app) as client:
        generated = client.post("/v1/completions", json={**generation().model_dump(),
                               "image": image_string(), "retain_decision_prefix": True}).json()
        holder.service.images.image_limit = 1
        result = client.post("/v1/decisions", json=decision(messages=None, prefix_id=generated["prefix_id"],
                                                          image=image_string()).model_dump())
        assert result.status_code == 422
        assert result.json()["detail"][0]["loc"] == ["body", "image"]


def test_engine_configuration_keeps_operator_options_and_shuts_down(monkeypatch):
    from fuzev_vllm.server import engine_context
    from vllm.engine import arg_utils
    from vllm.v1.engine import async_llm
    observed = {}
    engine = Engine()

    class Args:
        def __init__(self, **kwargs):
            observed.update(kwargs)
        def create_engine_config(self, **kwargs):
            return NS()

    monkeypatch.setattr(arg_utils, "AsyncEngineArgs", Args)
    monkeypatch.setattr(async_llm.AsyncLLM, "from_vllm_config", lambda *a, **k: engine)
    async def case():
        async with engine_context({"additional_config": {"fuzev": {"x": 1}, "other": 2}})() as result:
            assert result is engine
    asyncio.run(case())
    assert observed["additional_config"] == {"fuzev": {"x": 1, "enabled": True}, "other": 2}
    assert engine.shutdown_called


def test_cli_routes_operator_settings(monkeypatch, tmp_path):
    import fuzev_vllm.server as server
    import uvicorn
    engine_file = tmp_path / "engine.json"
    engine_file.write_text('{"model":"model"}')
    observed = {}
    def app(context, config, **kwargs):
        observed.update(config=config, **kwargs)
        return "app"
    monkeypatch.setattr(server, "create_app", app)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["fuzev-vllm", "--engine-config", str(engine_file),
                        "--model-name", "operator-model", "--reasoning-parser", "none", "--api-key", "secret"])
    server.main()
    assert observed["config"].reasoning_parser is None
    assert observed["config"].model_name == "operator-model"
    assert observed["api_key"] == "secret"


@pytest.mark.gpu
def test_running_engine_systemone():
    endpoint = os.environ.get("FUZEV_TEST_ENDPOINT")
    if not endpoint:
        pytest.skip("Set FUZEV_TEST_ENDPOINT to a running FuZev vLLM server")
    response = httpx.post(endpoint.rstrip("/") + "/v1/systemone",
                          json={"state": "The sky is blue.", "questions": {"truth": {"type": "noul"}}}, timeout=120)
    assert response.status_code == 200, response.text
    assert 0 <= response.json()["answers"]["truth"]["noul"] <= 1


def test_chat_accepts_standard_openai_fields():
    holder = application()
    with TestClient(holder.app) as client:
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "developer", "content": [{"type": "text", "text": "Be brief."}]},
                         {"role": "user", "content": "Hi"}],
            "n": 1, "user": "client-7", "stop": "END", "max_completion_tokens": 8,
            "tools": [], "tool_choice": "none", "response_format": {"type": "text"}})
        assert response.status_code == 200, response.text
        request, _, _ = holder.service.api.sampling_args
        assert request.stop == ["END"] and request.max_tokens == 8
        assert request.messages[0].role == "system" and request.messages[0].content == "Be brief."
