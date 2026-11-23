"""Image editing backends.

Editors are registered in EDITORS; each entry maps a CLI name to a loader
returning an object with .edit(image, prompt, num_inference_steps, seed).
To add a new editing model, implement a small Editor subclass and register it.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image

# Model cache: set HF_HOME yourself if the default (~/.cache/huggingface) is too small.


class Editor:
    """Uniform interface over diffusers editing pipelines."""

    name: str = ""

    def __init__(self, device: str):
        self.device = device
        self.dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = self._load()
        self.pipe.set_progress_bar_config(disable=True)

    def _load(self):
        raise NotImplementedError

    def _generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device=self.pipe.device).manual_seed(seed)

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 40,
        seed: int = 42,
    ) -> Image.Image:
        raise NotImplementedError


class _Img2ImgEditor(Editor):
    """Shared edit() for img2img-style pipelines (SD 2.1 / SDXL)."""

    strength: float = 0.75

    def edit(self, image, prompt, num_inference_steps=40, seed=42):
        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            strength=self.strength,
            generator=self._generator(seed),
        )
        return result.images[0]


class SD21Editor(_Img2ImgEditor):
    name = "sd21"
    model_id = "sd2-community/stable-diffusion-2-1"

    def _load(self):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id, torch_dtype=self.dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        return pipe.to(self.device)


class SDXLEditor(_Img2ImgEditor):
    name = "sdxl"
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def _load(self):
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLImg2ImgPipeline

        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(self.model_id, torch_dtype=self.dtype)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        return pipe.to(self.device)


class InstructPix2PixEditor(Editor):
    name = "ip2p"
    model_id = "timbrooks/instruct-pix2pix"

    def _load(self):
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id, torch_dtype=self.dtype, safety_checker=None
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        return pipe.to(self.device)

    def edit(self, image, prompt, num_inference_steps=40, seed=42):
        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=1.5,
            generator=self._generator(seed),
        )
        return result.images[0]


EDITORS: dict[str, type[Editor]] = {
    SD21Editor.name: SD21Editor,
    SDXLEditor.name: SDXLEditor,
    InstructPix2PixEditor.name: InstructPix2PixEditor,
}


def get_editor(name: str, device: str) -> Editor:
    if name not in EDITORS:
        raise ValueError(f"Unknown editor {name!r}; available: {sorted(EDITORS)}")
    return EDITORS[name](device)
