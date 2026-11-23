"""Metric computation utilities."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.csd import get_csd_encoder

DINO_PREPROCESS = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def pil_to_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def compute_psnr(img_a: Image.Image, img_b: Image.Image) -> float:
    a = pil_to_np(img_a)
    b = pil_to_np(img_b)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def compute_ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    a = pil_to_np(img_a)
    b = pil_to_np(img_b)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


class MetricComputer:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._lpips = None
        self._clip_model = None
        self._clip_preprocess = None
        self._dino_model = None
        self._dino_preprocess = None
        self._face_app = None

    def _get_lpips(self):
        if self._lpips is None:
            import lpips

            self._lpips = lpips.LPIPS(net="alex").to(self.device).eval()
        return self._lpips

    def _get_clip(self):
        if self._clip_model is None:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self._clip_model = model.to(self.device).eval()
            self._clip_preprocess = preprocess
        return self._clip_model, self._clip_preprocess

    def _get_dino(self):
        if self._dino_model is None:
            # DINOv2 ViT-B/14; the CLS-token cosine similarity is S(d) for objects.
            from transformers import Dinov2Model

            model = Dinov2Model.from_pretrained("facebook/dinov2-base")
            self._dino_model = model.to(self.device).eval()
            self._dino_preprocess = DINO_PREPROCESS
        return self._dino_model, self._dino_preprocess

    @torch.inference_mode()
    def _embed_clip(self, model, preprocess, image: Image.Image) -> torch.Tensor:
        tensor = preprocess(image).unsqueeze(0).to(self.device)
        feat = model.encode_image(tensor)
        return F.normalize(feat, dim=-1)

    @torch.inference_mode()
    def _embed_dino(self, model, preprocess, image: Image.Image) -> torch.Tensor:
        tensor = preprocess(image).unsqueeze(0).to(self.device)
        out = model(pixel_values=tensor)
        feat = out.last_hidden_state[:, 0]
        return F.normalize(feat, dim=-1)

    def compute_lpips(self, img_a: Image.Image, img_b: Image.Image) -> float:
        import lpips

        def to_tensor(img: Image.Image) -> torch.Tensor:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            arr = arr * 2.0 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        h = min(img_a.height, img_b.height)
        w = min(img_a.width, img_b.width)
        a = img_a.resize((w, h))
        b = img_b.resize((w, h))
        return float(self._get_lpips()(to_tensor(a), to_tensor(b)).item())

    def compute_clip_img_sim(self, img_a: Image.Image, img_b: Image.Image) -> float:
        model, preprocess = self._get_clip()
        ea = self._embed_clip(model, preprocess, img_a)
        eb = self._embed_clip(model, preprocess, img_b)
        return float((ea @ eb.T).item())

    def compute_dino_sim(self, img_a: Image.Image, img_b: Image.Image) -> float:
        model, preprocess = self._get_dino()
        ea = self._embed_dino(model, preprocess, img_a)
        eb = self._embed_dino(model, preprocess, img_b)
        return float((ea @ eb.T).item())

    @torch.inference_mode()
    def compute_clip_text_sim(self, image: Image.Image, text: str) -> float:
        import open_clip

        model, preprocess = self._get_clip()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        img_feat = self._embed_clip(model, preprocess, image)
        tokens = tokenizer([text]).to(self.device)
        txt_feat = F.normalize(model.encode_text(tokens), dim=-1)
        return float((img_feat @ txt_feat.T).item())

    def _get_face_app(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=int(self.device.split(":")[-1]) if "cuda" in self.device else -1)
            self._face_app = app
        return self._face_app

    def _arcface_embedding(self, app, image: Image.Image) -> np.ndarray | None:
        """ArcFace embedding via SCRFD detect+align, with center-crop fallback."""
        rgb = np.asarray(image.convert("RGB"))
        faces = app.get(rgb)
        if faces:
            return faces[0].embedding

        # CelebA-HQ portraits are tight, pre-centered crops; SCRFD often returns
        # zero boxes even at very low thresholds. Fall back to a center-square crop.
        rec = app.models.get("recognition")
        if rec is None:
            return None
        h, w = rgb.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = rgb[y0 : y0 + side, x0 : x0 + side]
        face_bgr = np.asarray(Image.fromarray(crop).resize((112, 112), Image.BILINEAR))[:, :, ::-1]
        return rec.get_feat(face_bgr).reshape(-1)

    def compute_arcface_similarity(self, original: Image.Image, edited: Image.Image) -> tuple[float | None, bool]:
        """ArcFace identity embedding cosine similarity via InsightFace."""
        app = self._get_face_app()
        emb_o = self._arcface_embedding(app, original)
        emb_e = self._arcface_embedding(app, edited)
        if emb_o is None or emb_e is None:
            return None, False
        sim = float(np.dot(emb_o, emb_e) / (np.linalg.norm(emb_o) * np.linalg.norm(emb_e) + 1e-8))
        return sim, True

    def compute_csd_style_similarity(self, original: Image.Image, edited: Image.Image) -> float:
        # Prefer the job device; CSD-ViT-L on CPU is ~minutes/image.
        device = self.device if "cuda" in str(self.device) else "cpu"
        encoder = get_csd_encoder(device)
        return encoder.style_cosine_similarity(original, edited)

    def compute_object_concept_scores(
        self,
        original: Image.Image,
        edited: Image.Image,
        concept_label: str,
    ) -> tuple[float, float, float]:
        """Return (clip_text_sim, dino_image_sim, concept_similarity).

        concept_similarity for objects is DINO image similarity only (not CLIP).
        """
        clip_score = self.compute_clip_text_sim(edited, f"a photo of {concept_label}")
        dino_score = self.compute_dino_sim(original, edited)
        return clip_score, dino_score, dino_score

    def compute_concept_metrics(
        self,
        dataset: str,
        original: Image.Image,
        edited: Image.Image,
        concept_label: str,
    ) -> dict[str, float | bool | None]:
        """Return concept scores.

        concept_similarity: faces=ArcFace, arts=CSD, objects=DINO image sim.
        clip_concept_sim is stored for objects but not used in concept_similarity.
        """
        out: dict[str, float | bool | None] = {
            "concept_similarity": math.nan,
            "clip_concept_sim": math.nan,
            "dino_concept_sim": math.nan,
            "face_detected": "",
        }
        if dataset == "faces":
            sim, detected = self.compute_arcface_similarity(original, edited)
            out["concept_similarity"] = sim if sim is not None else math.nan
            out["face_detected"] = detected
            return out
        if dataset == "arts":
            out["concept_similarity"] = self.compute_csd_style_similarity(original, edited)
            out["face_detected"] = True
            return out
        if dataset == "objects":
            clip_score, dino_score, combined = self.compute_object_concept_scores(
                original, edited, concept_label
            )
            out["clip_concept_sim"] = clip_score
            out["dino_concept_sim"] = dino_score
            out["concept_similarity"] = combined
            out["face_detected"] = True
            return out
        return out

    def compute_concept_similarity(
        self,
        dataset: str,
        original: Image.Image,
        edited: Image.Image,
        concept_label: str,
    ) -> tuple[float | None, bool]:
        metrics = self.compute_concept_metrics(dataset, original, edited, concept_label)
        sim = metrics["concept_similarity"]
        detected = metrics["face_detected"]
        if isinstance(sim, float) and math.isnan(sim):
            return None, bool(detected)
        return (float(sim) if sim is not None else None), bool(detected)

    def compute_pair_metrics(
        self,
        before: Image.Image,
        after: Image.Image,
    ) -> dict:
        return {
            "psnr": compute_psnr(before, after),
            "ssim": compute_ssim(before, after),
            "lpips": self.compute_lpips(before, after),
            "clip_img_sim": self.compute_clip_img_sim(before, after),
            "dino_sim": self.compute_dino_sim(before, after),
        }
