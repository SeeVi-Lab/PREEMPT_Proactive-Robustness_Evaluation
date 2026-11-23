"""Contrastive Style Descriptor (CSD) style-embedding utilities."""

from __future__ import annotations

import copy
from functools import lru_cache

from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from huggingface_hub import hf_hub_download
from PIL import Image

CSD_REPO = "tomg-group-umd/CSD-ViT-L"
CSD_INPUT_SIZE = 224

CSD_PREPROCESS = T.Compose(
    [
        T.Resize(size=CSD_INPUT_SIZE, interpolation=TF.InterpolationMode.BICUBIC),
        T.CenterCrop(CSD_INPUT_SIZE),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)


def convert_weights_float(model: nn.Module) -> None:
    """Convert applicable model parameters to fp32 (from CSD repo)."""

    def _convert_weights_to_fp32(layer: nn.Module) -> None:
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.float()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.float()

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)


def convert_state_dict(state_dict: dict) -> OrderedDict:
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "")
        new_state_dict[key] = value
    return new_state_dict


class CSD_CLIP(nn.Module):
    """CSD ViT-L style/content disentanglement model."""

    def __init__(self, name: str = "vit_large", content_proj_head: str = "default"):
        super().__init__()
        self.content_proj_head = content_proj_head
        if name == "vit_large":
            clipmodel, _ = clip.load("ViT-L/14")
            self.backbone = clipmodel.visual
            self.embedding_dim = 1024
        elif name == "vit_base":
            clipmodel, _ = clip.load("ViT-B/16")
            self.backbone = clipmodel.visual
            self.embedding_dim = 768
            self.feat_dim = 512
        else:
            raise ValueError(f"Unsupported CSD arch: {name}")

        convert_weights_float(self.backbone)
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        if content_proj_head == "custom":
            raise NotImplementedError("custom content projection head is not vendored")
        self.last_layer_content = copy.deepcopy(self.backbone.proj)
        self.backbone.proj = None

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def forward(self, input_data: torch.Tensor, alpha=None):
        feature = self.backbone(input_data)
        reverse_feature = feature if alpha is None else feature
        style_output = feature @ self.last_layer_style
        style_output = nn.functional.normalize(style_output, dim=1, p=2)
        content_output = reverse_feature @ self.last_layer_content
        content_output = nn.functional.normalize(content_output, dim=1, p=2)
        return feature, content_output, style_output


class CSDStyleEncoder:
    """Lazy-loaded CSD style embedding encoder."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model: CSD_CLIP | None = None

    def _get_model(self) -> CSD_CLIP:
        if self._model is None:
            model = CSD_CLIP("vit_large", "default")
            weight_path = hf_hub_download(CSD_REPO, "pytorch_model.bin")
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
            state_dict = convert_state_dict(checkpoint["model_state_dict"])
            model.load_state_dict(state_dict, strict=False)
            model.eval().to(self.device)
            self._model = model
        return self._model

    @torch.inference_mode()
    def embed_style(self, image: Image.Image) -> torch.Tensor:
        model = self._get_model()
        tensor = CSD_PREPROCESS(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled="cuda" in self.device):
            _, _, style = model(tensor)
        return style.squeeze(0)

    @torch.inference_mode()
    def style_cosine_similarity(self, img_a: Image.Image, img_b: Image.Image) -> float:
        ea = self.embed_style(img_a)
        eb = self.embed_style(img_b)
        return float(torch.dot(ea, eb).item())


@lru_cache(maxsize=4)
def get_csd_encoder(device: str = "cpu") -> CSDStyleEncoder:
    """Return a cached CSD encoder. Defaults to CPU to avoid VRAM clashes with CLIP/DINO."""
    return CSDStyleEncoder(device=device)
