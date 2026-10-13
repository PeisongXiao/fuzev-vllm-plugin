# Attach To An Existing vLLM Engine

Attach FuZev to the application's existing AsyncLLM, scheduler and cache.
Before constructing the engine, configure the
[vLLM fork](https://github.com/PeisongXiao/vllm-fuzev) with:

- `additional_config={"fuzev": {"enabled": True}}`
- `async_scheduling=False`
- `VLLM_USE_V2_MODEL_RUNNER=0`

## Python Calls

```python
from fuzev.api import validate_decision
from fuzev.service_contracts import GenerationRequest, ServiceDecisionRequest
from fuzev.systemone import SystemOneAdapter, SystemOneRequest
from fuzev_vllm.service import VllmServiceConfig, build_service

config = VllmServiceConfig(
    model_name="my-model", head_directory="/path/to/fuzev-head",
    head_device="cpu", calibration="raw",
)
service = await build_service(existing_async_llm, config)

request = ServiceDecisionRequest(
    primitive="choice",
    messages=[{"role": "user", "content": "I was charged twice."}],
    question="Which team should handle this ticket?",
    candidates=[
        {"id": "billing", "text": "Charges"},
        {"id": "shipping", "text": "Delivery"},
    ],
)
validate_decision(request, service.head)
decision = await service.decide(request, "req-1")
generation = await service.generate(
    GenerationRequest(
        messages=[{"role": "user", "content": "Say hello."}], max_tokens=32,
    ),
    "req-2",
)

adapter = SystemOneAdapter(
    service.decide, service.config.model_name,
    lambda text: len(service.tokenizer.encode(text, add_special_tokens=False, truncation=False)),
    head_id=service.head.head_id, max_options=service.head.max_options,
    primitives=service.head.primitives,
)
answer = await adapter.answer(SystemOneRequest.model_validate({
    "state": "I was charged twice.",
    "questions": {"refund": {
        "type": "noul", "instructions": "Does the customer request a refund?",
    }},
}))

# During application shutdown:
await service.close()
```

Make these calls on the application's event loop. The application manages the
engine lifecycle. Closing the service drains its jobs, releases retained
prefixes and closes the head.

For direct calls:

- Validate external inputs once.
- Decode the request's `image` string with `fuzev.images.decode_image` and pass
  the RGB image as `image=`. Close it after the awaited call completes.
- `GenerationRequest` parses a message's `image_url` content part into `image`.

HTTP routes perform this preprocessing and cleanup.

## HTTP Routes

`create_app(engine_context, config, *, api_key=None, max_request_bytes=None)`
takes a zero-argument function that returns an async context manager yielding
the engine. `max_request_bytes=None` keeps the 8 MiB request body default:

```python
from contextlib import asynccontextmanager
from fuzev_vllm.server import create_app

@asynccontextmanager
async def engine_context():
    yield existing_async_llm

app = create_app(engine_context, config, api_key=None)
```

The app lifespan loads the head, mounts SystemOne and generation routes, drains
HTTP work and closes the service. The context function above leaves the engine
running for the application.

Service behavior:

- Generation template arguments combine startup settings with request
  `chat_template_kwargs` for messages; requests take precedence. Raw `prompt`
  requests ignore them.
- With a reasoning parser that matches the model (default `qwen3`), thinking
  appears in `message.reasoning` and answer text in `message.content`.
- Retained prefixes keep accepted tokens and image payloads for later
  decisions.
- The head bundle supplies feature width, maximum option count and decision
  types (`primitives` in `config.json`). The service checks feature geometry and
  encoding compatibility.

## Limitations

- FuZev v1.0 is an experimental research release.
- v1.0 accepts 2 to 16 Choice options or Score levels per question. The head
  bundle declares the maximum (`max_options` in its `config.json`); the
  published head declares 16. Larger counts are planned for v1.1. Hosted Jev
  and Kev allow different counts; see the
  [capacity comparison](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#capacity-comparison).
- Noul has exactly two outcomes.
- The [API limitations](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#limitations)
  list the supported topology and experimental image scope.

Measurements are in
[integration results](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md);
research methods are in the [archive](https://github.com/PeisongXiao/fuzev-research-archive).
