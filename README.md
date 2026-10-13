# FuZev vLLM Plugin

Serve generation and Jev-style Choice, Noul and Score decisions through the
[vllm-fuzev](https://github.com/PeisongXiao/vllm-fuzev) fork. Decisions use
`POST /v1/systemone`, the SystemOne API of hosted Jev and
[Kev](https://github.com/jaredpalmer/kev).

The plugin loads the [decision head](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev)
with [fuzev-runtime](https://github.com/PeisongXiao/fuzev-runtime) and attaches it
to the engine's scheduler and cache. The
[API guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md) covers
requests, responses and SDK examples.

[FuZev index](https://github.com/PeisongXiao/fuzev)

**Contents**

- [Install And Serve](#install-and-serve)
- [Configuration](#configuration)
- [Tests](#tests)
- [Results](#results)
- [License](#license)
- [Limitations](#limitations)

## Install And Serve

1. Follow the canonical
   [environment and engine installation](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#create-an-environment),
   including the [prebuilt FlashInfer kernels](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#vllm).
2. Run the [vLLM NVFP4 integration step](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#vllm-with-nvfp4).
   For EXL3, use the [vLLM EXL3 steps](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#vllm-with-exl3),
   including the optional AGPL-3.0-only
   [quantization plugin](https://github.com/PeisongXiao/vllm-exl3-plugin).
3. Keep the same environment active for startup.

Create `engine.json` from standard vLLM `AsyncEngineArgs`, for example:

```json
{
  "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
  "async_scheduling": false,
  "gpu_memory_utilization": 0.75,
  "max_model_len": 32768,
  "max_num_seqs": 8,
  "additional_config": {"ple_storage": "mmap"}
}
```

- The memory fraction, context window and sequence count are example GB10
  operator settings.
- `ple_storage: mmap` reads Qwen3.8's per-layer n-gram embedding tables from the
  memory-mapped checkpoint and selects PIECEWISE CUDA graphs.
- Decision readout uses synchronous scheduling (`async_scheduling: false`) and
  the V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`).

Place the head bundle in `fuzev-head`,
[verify its files](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#head-and-model-files),
then start:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
fuzev-vllm --engine-config engine.json --model-name my-model \
  --head-directory ./fuzev-head --head-device cpu
```

`python -m fuzev_vllm` is equivalent. For packed EXL3, set
`VLLM_PLUGINS=vllm_exl3` as the canonical guide describes.

Send a question:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' -d '{
    "state": {"message": "Please refund the duplicate charge."},
    "questions": {"refund": {
      "type": "noul", "instructions": "Does the customer request a refund?"
    }}
  }'
```

The response identifies the configured model and includes
`answers.refund.noul` (the Yes probability), `usage` and `latency_ms`.

## Configuration

Put model selection, context, memory, scheduling and speculation settings in
`engine.json`.

Omitted request sampling fields use the model's `generation_config.json`
defaults, as in `vllm serve`. An omitted `max_tokens` runs to the remaining context
or the model's default output length, whichever is shorter.

| Option | Default | Purpose |
| --- | --- | --- |
| `--engine-config` | Required | JSON file of `AsyncEngineArgs` |
| `--model-name` | Required | Advertised serving model ID |
| `--head-directory` | Unset | Decision head bundle; omit for generation only |
| `--max-request-bytes` | `8388608` (8 MiB) | Request body limit, including base64 images |
| `--host`, `--port` | `127.0.0.1`, `8000` | Listener address |
| `--api-key` | Unset | Bearer authentication |
| `--head-device` | `cpu` | Pass `cuda` to score on the GPU |
| `--calibration` | `raw` | Use `calibrated` for bundle temperatures |
| `--prefix-capacity` | `128` | Retained-prefix count; LRU eviction, no time-based expiry |
| `--reasoning-parser` | `qwen3` | vLLM reasoning parser name, or `none` |
| `--chat-template-kwargs` | `{}` | Generation template arguments |

For example, `--chat-template-kwargs '{"enable_thinking":false}'` disables
thinking by default. Request `chat_template_kwargs` takes precedence.

Reasoning parsers:

- The default `qwen3` parser suits Qwen3-style thinking models. For other
  models, pass their vLLM parser name or `none`.
- With a matching parser, thinking appears in `message.reasoning` and the answer
  in `message.content`.
- With `qwen3`, output without `</think>` returns `content: null`.
- An unknown parser name stops startup with vLLM's list of available parsers.

To embed the plugin in an application, see
[Python attachment](docs/attachment.md). Closing the service releases its
resources; the application keeps its engine.

## Tests

From this repository, with the runtime and engine installed:

```bash
python -m pip install '.[test]'
python -m pytest
```

The suite covers public API behavior, the TypeSafe SDK, reasoning parsing, image
inputs, concurrent decisions and cancellation.

Two opt-in checks skip unless their variables are set:

- `FUZEV_TEST_TOKENIZER` names a Qwen `tokenizer.json` for the image-boundary
  tokenization check.
- `FUZEV_TEST_ENDPOINT` names a running server, started without `--api-key`,
  for a check that sends one SystemOne request:

```bash
FUZEV_TEST_ENDPOINT=http://127.0.0.1:8000 python -m pytest -m gpu
```

## Results

[Benchmark results](https://github.com/PeisongXiao/fuzev/blob/main/docs/benchmarks.md)
and [integration results](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md)
report measured quality and generation overhead.

## License

This plugin uses [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

## Limitations

- FuZev v1.0 is an experimental research release.
- v1.0 accepts 2 to 16 Choice options or Score levels per question. The head
  bundle declares the maximum (`max_options` in its `config.json`); the
  published head declares 16. Larger counts are planned for v1.1. Hosted Jev
  and Kev allow different counts; see the
  [capacity comparison](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#capacity-comparison).
- Noul has exactly two outcomes.
- Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130). The engine needs
  CUDA; the head runs on CPU or CUDA (`--head-device`).
- Images are experimental. Image decisions use the head's Qwen-format encoding.
  Generation requests with an image use the served model's own image
  placeholder.
- Only CPU tests cover image requests; v1.0 reports no GPU image run.
- EXL3 checkpoints with MTP are unsupported in v1.0.
- The [engine limitations](https://github.com/PeisongXiao/vllm-fuzev/blob/main/docs/design/fuzev.md#limitations)
  list vLLM's supported decision topology. The
  [API limitations](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#limitations)
  give the full list.

Research methods are in the [archive](https://github.com/PeisongXiao/fuzev-research-archive).
