"""Generation template options and model-specific image markers on CPU."""

import asyncio
from types import SimpleNamespace as NS

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from fuzev.encoding import TEMPLATE
from fuzev.images import decode_image
from fuzev_vllm.service import VllmAPI, VllmService
from fixtures import API, Engine, config, decision, generation, image_string, service


def use_real_template(svc, template):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        chat_template=template)
    svc.tokenizer.apply_chat_template = tokenizer.apply_chat_template


def test_operator_and_request_kwargs_change_actual_generation_rendering():
    async def case():
        svc = service(generation_chat_template_kwargs={"enable_thinking": False})
        use_real_template(svc, TEMPLATE.read_text())
        try:
            messages = [{"role": "user", "content": "Question"}]
            await svc.generate(generation(prompt=None, messages=messages), "operator")
            operator_tokens = svc.engine.calls[-1][1]
            assert svc.tokenizer.decode(operator_tokens).endswith("<think>\n\n</think>\n\n")
            await svc.generate(generation(prompt=None, messages=messages,
                                          chat_template_kwargs={"enable_thinking": True}), "request")
            request_tokens = svc.engine.calls[-1][1]
            assert svc.tokenizer.decode(request_tokens).endswith("<think>\n")
            assert request_tokens != operator_tokens
        finally:
            await svc.close()
    asyncio.run(case())


def test_generation_kwargs_leave_actual_decision_tokens_unchanged():
    async def case():
        svc = service()
        use_real_template(svc, TEMPLATE.read_text())
        messages = [{"role": "user", "content": "Earlier question"},
                    {"role": "assistant", "content": "<think>Earlier reasoning</think>Earlier answer"},
                    {"role": "user", "content": "Question"}]
        request = decision(messages=messages)
        try:
            before = await svc.decide(request, "before")
            original = [tokens for kind, tokens, _, _ in svc.engine.calls if kind == "encode"]
            assert "Earlier reasoning" in svc.tokenizer.decode(original[0])
            svc.config.generation_chat_template_kwargs = {
                "enable_thinking": True, "reasoning_effort": "high", "preserve_thinking": False}
            await svc.generate(generation(prompt=None, messages=messages, chat_template_kwargs={
                "enable_thinking": False, "reasoning_effort": "low", "preserve_thinking": False}), "chat")
            after = await svc.decide(request, "after")
            submitted = [tokens for kind, tokens, _, _ in svc.engine.calls if kind == "encode"]
            assert submitted[len(original):] == original
            assert after["scores"] == before["scores"]
        finally:
            await svc.close()
    asyncio.run(case())


@pytest.mark.parametrize("form,expected", [
    ("prompt", "<image>Question"),
    ("messages", "USER: <image>\nQuestion\nASSISTANT: "),
    ("system-only", "SYSTEM: Policy\nUSER: <image>\n\nASSISTANT: "),
])
def test_image_generation_uses_non_qwen_model_and_template_placeholder(form, expected):
    class ImageModel:
        @staticmethod
        def get_placeholder_str(modality, index):
            assert (modality, index) == ("image", 1)
            return "<image>"

    template = ("{% for message in messages %}{{ message.role|upper }}: "
                "{% if message.content is string %}{{ message.content }}{% else %}"
                "{% for part in message.content %}{% if part.type == 'image' %}<image>\n"
                "{% else %}{{ part.text }}{% endif %}{% endfor %}{% endif %}{{ '\n' }}{% endfor %}"
                "{% if add_generation_prompt %}ASSISTANT: {% endif %}")

    async def case():
        engine, api = Engine(), API()
        model = engine.vllm_config.model_config
        model.model = f"image-placeholder-{form}"
        model.convert_type, model.runner_type = "none", "generate"
        model.trust_remote_code, model.model_impl = False, "vllm"
        model.hf_config = NS(architectures=["FixtureImageModel"])
        model._get_transformers_backend_cls = lambda: "TransformersForCausalLM"

        def resolve(architectures, *, model_config):
            assert model_config is model and architectures == ["FixtureImageModel"]
            return ImageModel, "FixtureImageModel"

        # Keep the upstream resolver real, including its ModelConfig field access.
        model.registry = NS(resolve_model_cls=resolve)
        engine.tokenizer.markers, engine.tokenizer.marker_ids = ("<image>",), (150002,)
        api.image_placeholder = VllmAPI.image_placeholder
        svc = VllmService(engine, engine.tokenizer, None, config(head_directory=None), api=api)
        use_real_template(svc, template)
        request = (generation(prompt="Question") if form == "prompt" else generation(prompt=None, messages=[
            {"role": "system", "content": "Policy"} if form == "system-only" else
            {"role": "user", "content": "Question"}]))
        try:
            with decode_image(image_string()) as image:
                result = await svc.generate(request, "image", image=image)
            assert result["choices"][0]["finish_reason"] == "stop"
            raw, _ = engine.renderer.calls[0]
            assert raw == tuple(engine.tokenizer.encode(expected))
            submitted = engine.calls[0][1]
            assert submitted.count(150002) == 4
            assert 150001 not in submitted and 150003 not in submitted
        finally:
            await svc.close()
    asyncio.run(case())
