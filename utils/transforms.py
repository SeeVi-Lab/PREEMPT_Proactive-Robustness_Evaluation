"""Level 1 physical image transformations."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# Protected images are plain-stretched to 512x512; clean inputs must match so
# resolution is not a confound (and so huge originals don't stall the editors).
EDIT_RESOLUTION = (512, 512)


def load_edit_input(path: Path) -> Image.Image:
    image = load_rgb(path)
    if image.size != EDIT_RESOLUTION:
        image = image.resize(EDIT_RESOLUTION, Image.Resampling.LANCZOS)
    return image


def save_rgb(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image.save(path, quality=95)
    else:
        image.save(path)


def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_crop_resize(image: Image.Image, crop_ratio: float) -> Image.Image:
    w, h = image.size
    new_w = max(1, int(w * crop_ratio))
    new_h = max(1, int(h * crop_ratio))
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)


def apply_brightness_contrast(
    image: Image.Image, brightness: float, contrast: float
) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(brightness)
    return ImageEnhance.Contrast(image).enhance(contrast)


def apply_horizontal_flip(image: Image.Image) -> Image.Image:
    return ImageOps.mirror(image)


def apply_transform(image: Image.Image, transform: str, config: dict) -> Image.Image:
    cfg = config.get(transform, {})
    ttype = cfg.get("type", transform)

    if ttype == "jpeg":
        return apply_jpeg(image, cfg["quality"])
    if ttype == "gaussian_blur":
        return apply_gaussian_blur(image, cfg["radius"])
    if ttype == "crop_resize":
        return apply_crop_resize(image, cfg["crop_ratio"])
    if ttype == "brightness_contrast":
        return apply_brightness_contrast(image, cfg["brightness"], cfg["contrast"])
    if ttype == "horizontal_flip":
        return apply_horizontal_flip(image)
    raise ValueError(f"Unknown transform: {transform}")
