"""Retained-prefix image preparation preserves the fixed decision encoding."""

import os
from collections import Counter

import pytest
from fastapi.testclient import TestClient

from fuzev.encoding import IMAGE_PLACEHOLDER, candidate_suffix, query_suffix
from fuzev.systemone import SystemOneRequest, decision_request, state_text

from fixtures import application, generation, image_string


@pytest.mark.parametrize("retained_image", [False, True])
def test_retained_prefix_new_image_prepared_once_with_standalone_parity(retained_image):
    holder = application()
    with TestClient(holder.app) as client:
        generated = client.post("/v1/completions", json=generation(
            retain_decision_prefix=True,
            image=image_string() if retained_image else None).model_dump(mode="json"))
        assert generated.status_code == 200, generated.text
        prefix = holder.service.prefixes.get(generated.json()["prefix_id"])
        original_tokens = prefix.token_ids
        request = SystemOneRequest.model_validate({
            "state": {"detail": "new evidence\nsecond line"}, "prefix_id": prefix.id,
            "image": image_string(color=(90, 80, 70)),
            "questions": {
                "pick": {"type": "choice", "question": "Which?", "criteria": {"a": "first", "b": "second"}},
                "check": {"type": "noul", "criteria": {"true": "Urgent", "false": "Not urgent"}},
            },
        })
        engine = holder.engine
        start, render_start = len(engine.inputs), len(engine.renderer.calls)
        result = client.post("/v1/systemone", json=request.model_dump(mode="json"))
        assert result.status_code == 200, result.text
        assert len(engine.renderer.calls) == render_start + 1
        batch_inputs = engine.inputs[start:]
        expected = Counter()
        total = 0
        for key, question in request.questions.items():
            direct = decision_request(request, question, "fixture-only", "reference", state_text(request.state))
            candidates = [item.model_dump() for item in direct.candidates]
            full_suffix = query_suffix(direct.question, candidates, IMAGE_PLACEHOLDER + "\n" + direct.additional_context)
            raw = tuple(engine.tokenizer.encode(full_suffix))
            pad = raw.index(150002)
            # Expected tokens come from encoding the full suffix in one piece.
            query = original_tokens + raw[:pad] + (150002,) * 4 + raw[pad + 1:]
            prompts = [query] + [query + tuple(engine.tokenizer.encode(candidate_suffix(c))) for c in candidates]
            expected.update(prompts)
            start, render_start = len(engine.inputs), len(engine.renderer.calls)
            response = client.post("/v1/decisions", json=direct.model_dump(mode="json"))
            assert response.status_code == 200, response.text
            assert len(engine.renderer.calls) == render_start + 1
            standalone = engine.inputs[start:]
            assert Counter(tuple(p["prompt_token_ids"]) for p in standalone) == Counter(prompts)
            body = response.json()
            assert [row["id"] for row in body["scores"]] == [candidate["id"] for candidate in candidates]
            probabilities = [row["calibrated_probability"] for row in body["scores"]]
            answer = result.json()["answers"][key]
            if question.type == "choice":
                assert answer["probabilities"] == dict(zip(question.criteria, probabilities))
                assert answer["choice"] == body["selected_id"]
            else:
                assert answer["noul"] == probabilities[0]
            total += body["execution"]["submitted_tokens"]
        assert Counter(tuple(p["prompt_token_ids"]) for p in batch_inputs) == expected
        assert result.json()["usage"]["input_tokens"] == total
        assert prefix.token_ids == original_tokens


def test_real_qwen_image_boundary_preserves_unsplit_encoding():
    path = os.environ.get("FUZEV_TEST_TOKENIZER")
    if not path:
        pytest.skip("Set FUZEV_TEST_TOKENIZER to an already cached Qwen tokenizer.json")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(path)
    candidates = [{"text": "first"}, {"text": "second\n\u4e8c"}]
    for question in ("", "Which?", "\u4e8c?\n"):
        for additional in ("", " leading\ntrailing ", "\u56fe\u50cf", IMAGE_PLACEHOLDER):
            full = query_suffix(question, candidates, IMAGE_PLACEHOLDER + "\n" + additional)
            header, marker, tail = full.partition(IMAGE_PLACEHOLDER)
            common = tokenizer.encode(header + marker, add_special_tokens=False).ids
            assert common[-1] == tokenizer.token_to_id("<|vision_end|>")
            assert tokenizer.encode(full, add_special_tokens=False).ids == (
                common + tokenizer.encode(tail, add_special_tokens=False).ids)
