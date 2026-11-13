"""
Pre-trained AI-content detectors from the Hugging Face Hub.

They are NOT downloaded automatically. Run `python download_models.py` once in the
backend folder; it saves them to models/pretrained/, and the backend loads them from
there (local_files_only) and never downloads anything at runtime. A copy already in the
Hugging Face cache is used if the project folder doesn't have one.

Pick other models with the TEXT_DETECTOR_MODEL / IMAGE_DETECTOR_MODEL environment
variables (a Hub id or a local folder), or set them to "none" to disable.
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import (AutoConfig, AutoModel, AutoModelForImageClassification,
                          AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel)
from transformers.utils import cached_file

DEFAULT_TEXT_MODEL = "Oxidane/tmr-ai-text-detector"
DEFAULT_IMAGE_MODEL = "haywoodsloan/ai-image-detector-deploy"
DEFAULT_VIDEO_MODEL = "ducbao210/video-deepfake-detection"
BATCH_SIZE = 16

# Label names that mean "AI-generated" in the models' id2label
AI_LABEL_HINTS = ("ai", "chatgpt", "gpt", "machine", "generated", "fake", "artificial", "synthetic")


def configured_model(env_var: str, default: str):
    value = os.getenv(env_var, default).strip()
    return None if value.lower() in ("", "none", "off") else value


TEXT_MODEL_ID = configured_model("TEXT_DETECTOR_MODEL", DEFAULT_TEXT_MODEL)
IMAGE_MODEL_ID = configured_model("IMAGE_DETECTOR_MODEL", DEFAULT_IMAGE_MODEL)
VIDEO_MODEL_ID = configured_model("VIDEO_DETECTOR_MODEL", DEFAULT_VIDEO_MODEL)

PRETRAINED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "pretrained"))


def model_path(model_id: str) -> str:
    """Where download_models.py saves a Hub model, e.g. models/pretrained/Oxidane--tmr-ai-text-detector."""
    return os.path.join(PRETRAINED_DIR, model_id.replace("/", "--"))


def model_source(model_id: str) -> str:
    """The project copy if downloaded; otherwise the id itself (a local folder, or the Hugging Face cache)."""
    path = model_path(model_id)
    return path if os.path.isdir(path) else model_id


class NotDownloadedError(RuntimeError):
    pass


def ai_label_index(config) -> int:
    for idx, label in (config.id2label or {}).items():
        if str(label).lower().startswith(AI_LABEL_HINTS):
            return int(idx)
    return 1


class DesklibAIDetectionModel(PreTrainedModel):
    """Custom head used by desklib/ai-text-detector-* (mean pooling + one logit), as in its model card."""
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        hidden = self.model(input_ids, attention_mask=attention_mask)[0]
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


class TextDetector:
    """A Hub text classifier that outputs P(AI-generated) for a batch of texts."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        source = model_source(model_id)
        try:
            config = AutoConfig.from_pretrained(source, local_files_only=True)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
            except Exception:
                # tokenizer.json saved by a newer `tokenizers` than transformers 4.40 allows can't be
                # parsed here; build the tokenizer from vocab.json/merges.txt instead.
                self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=False)
            self.desklib = "DesklibAIDetectionModel" in (config.architectures or [])
            model_cls = DesklibAIDetectionModel if self.desklib else AutoModelForSequenceClassification
            self.model = model_cls.from_pretrained(source, local_files_only=True).eval()
        except OSError as e:
            raise NotDownloadedError(model_id) from e
        self.single_logit = self.desklib or config.num_labels == 1
        self.ai_index = ai_label_index(config)
        self.max_length = 768 if self.desklib else 512

    def predict(self, texts) -> np.ndarray:
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            enc = self.tokenizer(list(texts[i:i + BATCH_SIZE]), return_tensors="pt", truncation=True,
                                 max_length=self.max_length, padding=True)
            with torch.no_grad():
                if self.desklib:
                    logits = self.model(enc["input_ids"], attention_mask=enc["attention_mask"])
                else:
                    logits = self.model(**enc).logits
            if self.single_logit:
                out.append(torch.sigmoid(logits[:, 0]).numpy())
            else:
                out.append(torch.softmax(logits, dim=-1)[:, self.ai_index].numpy())
        return np.concatenate(out)


class ImageDetector:
    """A Hub image classifier that outputs P(AI-generated) for an image file.
    Preprocessing is read from the model's preprocessor_config.json and done here, so it
    does not depend on the installed transformers version's image processors."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        source = model_source(model_id)
        try:
            self.model = AutoModelForImageClassification.from_pretrained(source, local_files_only=True).eval()
            with open(cached_file(source, "preprocessor_config.json", local_files_only=True)) as f:
                prep = json.load(f)
        except OSError as e:
            raise NotDownloadedError(model_id) from e
        config = self.model.config
        self.single_logit = config.num_labels == 1
        self.ai_index = ai_label_index(config)

        self.size = prep.get("size", {"height": 224, "width": 224})
        if isinstance(self.size, int):
            self.size = {"shortest_edge": self.size}
        crop = prep.get("crop_size")
        self.crop = (crop["height"], crop["width"]) if isinstance(crop, dict) else (crop, crop) if crop else None
        self.do_crop = prep.get("do_center_crop", False) and self.crop is not None
        self.resample = Image.Resampling(prep.get("resample", Image.Resampling.BICUBIC))
        self.rescale = prep.get("rescale_factor", 1 / 255)
        self.mean = np.array(prep.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.std = np.array(prep.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32)

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        w, h = img.size
        if "shortest_edge" in self.size:
            scale = self.size["shortest_edge"] / min(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), self.resample)
        else:
            img = img.resize((self.size["width"], self.size["height"]), self.resample)
        if self.do_crop:
            ch, cw = self.crop
            left, top = (img.width - cw) // 2, (img.height - ch) // 2
            img = img.crop((left, top, left + cw, top + ch))
        arr = (np.asarray(img, dtype=np.float32) * self.rescale - self.mean) / self.std
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    def predict(self, filepath: str) -> float:
        with Image.open(filepath) as img:
            pixels = self._preprocess(img)
        with torch.no_grad():
            logits = self.model(pixel_values=pixels).logits
        if self.single_logit:
            return float(torch.sigmoid(logits[0, 0]))
        return float(torch.softmax(logits, dim=-1)[0, self.ai_index])


class _VideoLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return torch.nn.functional.layer_norm(x.permute(0, 2, 3, 1), self.weight.shape,
                                              self.weight, self.bias).permute(0, 3, 1, 2)


class _VideoBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels) * 1e-6)
        self.conv_dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(channels, channels * 4)
        self.mlp.fc2 = nn.Linear(channels * 4, channels)

    def forward(self, x):
        residual = x
        x = self.conv_dw(x).permute(0, 2, 3, 1)
        x = self.mlp.fc2(torch.nn.functional.gelu(self.mlp.fc1(self.norm(x)))) * self.gamma
        return (x.permute(0, 3, 1, 2) + residual)


class _VideoStage(nn.Module):
    def __init__(self, channels, depth, previous_channels=None):
        super().__init__()
        if previous_channels:
            self.downsample = nn.Sequential(_VideoLayerNorm(previous_channels),
                                            nn.Conv2d(previous_channels, channels, 2, stride=2))
        self.blocks = nn.Sequential(*[_VideoBlock(channels) for _ in range(depth)])

    def forward(self, x):
        if hasattr(self, "downsample"):
            x = self.downsample(x)
        return self.blocks(x)


class VideoConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.stem = nn.Sequential(nn.Conv2d(3, 96, 4, stride=4), _VideoLayerNorm(96))
        self.encoder.stages = nn.ModuleList()
        depths = (3, 3, 27, 3)
        dims = (96, 192, 384, 768)
        for index, (channels, depth) in enumerate(zip(dims, depths)):
            self.encoder.stages.append(_VideoStage(channels, depth, dims[index - 1] if index else None))
        self.encoder.head = nn.Module()
        self.encoder.head.norm = nn.LayerNorm(768)
        self.classifier = nn.Sequential(nn.Dropout(0.0), nn.Linear(768, 2))

    def forward(self, x):
        x = self.encoder.stem(x)
        for stage in self.encoder.stages:
            x = stage(x)
        x = self.encoder.head.norm(x.mean(dim=(2, 3)))
        return self.classifier(x)


class VideoDetector:
    """Frame-level adapter for the ConvNeXt checkpoint published by the video model."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.checkpoint_url = (
            "https://huggingface.co/ducbao210/video-deepfake-detection/"
            "resolve/main/checkpoints/convnext/best.pth"
        )
        source = model_path(model_id)
        checkpoint = os.path.join(source, "convnext-best.pth")
        if not os.path.isfile(checkpoint):
            raise NotDownloadedError(model_id)
        try:
            self.model = VideoConvNeXt()
            state = torch.load(checkpoint, map_location="cpu")
            state = state.get("model_state_dict", state.get("state_dict", state))
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
            self.size = {"height": 224, "width": 224}
            self.crop = None
            self.do_crop = False
            self.resample = Image.Resampling.BICUBIC
            self.rescale = 1 / 255
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        except (OSError, RuntimeError, KeyError, TypeError) as e:
            raise RuntimeError(f"Could not load video checkpoint {checkpoint}: {e}") from e

    def predict_frames(self, frames) -> float:
        pixels = torch.stack([ImageDetector._preprocess(self, frame).squeeze(0) for frame in frames])
        with torch.no_grad():
            logits = self.model(pixels)
        return float(torch.softmax(logits, dim=-1)[:, 1].mean())


def load_detector(cls, model_id, service_name):
    """Load a pre-trained detector, or return None with a hint if it isn't downloaded."""
    if not model_id:
        return None
    try:
        detector = cls(model_id)
        print(f"[{service_name}] Pre-trained detector loaded: {model_id}")
        return detector
    except NotDownloadedError:
        print(f"[{service_name}] Pre-trained detector {model_id} is not downloaded — "
              f"run `python download_models.py` in the backend folder to enable it.")
    except Exception as e:
        print(f"[{service_name}] Could not load pre-trained detector {model_id}: {e}")
    return None
