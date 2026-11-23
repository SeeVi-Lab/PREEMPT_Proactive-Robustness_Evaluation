"""Watermark interface for PREEMPT.

PREEMPT does not ship any watermarking method. To evaluate a watermark, wrap the
method's own (official) decoder in a class that follows ``WatermarkDecoder`` and
pass it to ``scripts/score_watermark.py``.

For watermark-based methods the paradigm-specific metric S(d) is the message
retrieval bit accuracy (paper Sec. 3.2), so the score is written both as
``bit_accuracy`` and as ``concept_similarity`` (the column the survival
analysis reads).
"""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class WatermarkDecoder(Protocol):
    """Decode the embedded message of one image as a bit string, e.g. ``"0110..."``."""

    def decode(self, image: Image.Image) -> str: ...


def bit_accuracy(decoded_bits: str, reference_bits: str) -> float:
    """Fraction of payload bits recovered correctly.

    Missing bits (decoded string shorter than the payload) count as wrong.
    """
    if not reference_bits or set(reference_bits) - {"0", "1"}:
        raise ValueError("reference_bits must be a non-empty string of 0/1")
    correct = sum(a == b for a, b in zip(decoded_bits, reference_bits))
    return correct / len(reference_bits)


def text_to_bits(message: str, encoding: str = "utf-8") -> str:
    """Helper for text payloads: each byte -> 8 bits (MSB first)."""
    return "".join(f"{byte:08b}" for byte in message.encode(encoding))


def load_decoder(spec: str, **kwargs) -> WatermarkDecoder:
    """Load a decoder from ``"package.module:ClassName"`` and instantiate it."""
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"decoder spec must look like 'module:ClassName', got {spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    decoder = cls(**kwargs)
    if not isinstance(decoder, WatermarkDecoder):
        raise TypeError(f"{spec} has no decode(image) -> str method")
    return decoder
