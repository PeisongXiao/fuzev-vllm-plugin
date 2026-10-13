"""Image preprocessing through the existing vLLM renderer."""

import asyncio
import time
from dataclasses import dataclass, replace

from fuzev.api import validation_error


@dataclass(frozen=True)
class ImagePayload:
    item: object
    mm_hash: str
    placeholder: object

    def shifted(self, offset):
        return replace(self, placeholder=replace(self.placeholder, offset=self.placeholder.offset + offset))


@dataclass(frozen=True)
class MultimodalPayload:
    images: tuple[ImagePayload, ...]

    def append(self, suffix, offset):
        return MultimodalPayload(self.images + tuple(image.shifted(offset) for image in suffix.images))

    def engine_input(self, tokens):
        return {"type": "multimodal", "prompt_token_ids": list(tokens),
                "mm_kwargs": {"image": [image.item for image in self.images]},
                "mm_hashes": {"image": [image.mm_hash for image in self.images]},
                "mm_placeholders": {"image": [image.placeholder for image in self.images]}}


class VllmImages:
    def __init__(self, renderer, placeholder):
        self.renderer = renderer
        self.image_limit = renderer.mm_processor.info.allowed_mm_limits.get("image", 0)
        self.placeholder = placeholder or ""

    def check_count(self, payload):
        if len(payload.images) > self.image_limit:
            raise validation_error("image", f"The engine allows {self.image_limit} images per prompt")

    async def _prepare(self, image, raw_tokens):
        prompt = {"prompt_token_ids": list(raw_tokens), "multi_modal_data": {"image": [image]},
                  "mm_processor_kwargs": {"truncation": False}}
        # skip_mm_cache returns the processed image data itself, which retained prefixes reuse.
        processed = await self.renderer.process_for_engine_async(prompt, time.time(), skip_mm_cache=True)
        item, = processed["mm_kwargs"]["image"]
        mm_hash, = processed["mm_hashes"]["image"]
        position, = processed["mm_placeholders"]["image"]
        tokens = tuple(processed["prompt_token_ids"])
        if item is None:
            raise RuntimeError("Image preprocessing returned a cache handle without image data")
        return tokens, MultimodalPayload((ImagePayload(item, mm_hash, position),))

    async def prepare(self, image, raw_tokens):
        task = asyncio.create_task(self._prepare(image, raw_tokens))
        interrupted = False
        # The renderer may be running a CPU executor. Keep its image alive until
        # it finishes, even when a client cancels repeatedly.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                interrupted = True
            except Exception:
                break
        try:
            result = task.result()
        finally:
            if interrupted:
                raise asyncio.CancelledError
        return result
