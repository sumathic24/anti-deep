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
import cv2
try:
    import tensorflow as tf
except ImportError:
    tf = None
from PIL import Image
from transformers import (AutoConfig, AutoImageProcessor, AutoModel, AutoModelForImageClassification,
                          AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel)
from transformers.utils import cached_file

DEFAULT_TEXT_MODEL = "desklib/ai-text-detector-v1.01"
DEFAULT_IMAGE_MODEL = "dima806/deepfake_vs_real_image_detection"
DEFAULT_VIDEO_MODEL = "DarkVision/Deepfake_detection_video"
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
            self.processor = AutoImageProcessor.from_pretrained(source)
            self.model = AutoModelForImageClassification.from_pretrained(source).eval()
        except OSError as e:
            raise NotDownloadedError(model_id) from e
        config = self.model.config
        self.single_logit = config.num_labels == 1
        self.ai_index = ai_label_index(config)
        self.fake_index = int(config.label2id.get("Fake", self.ai_index))
        self.target_layer = self.model.vit.encoder.layer[-1].layernorm_after

    def _inputs(self, img: Image.Image):
        return self.processor(images=img.convert("RGB"), return_tensors="pt")

    def predict_with_gradcam(self, filepath: str):
        with Image.open(filepath) as img:
            inputs = self._inputs(img)
        self.model.zero_grad(set_to_none=True)
        activations = []
        gradients = []
        self.target_layer.register_forward_hook(lambda _, __, output: activations.append(output))
        self.target_layer.register_full_backward_hook(lambda _, __, output: gradients.append(output[0]))
        logits = self.model(**inputs).logits
        score = torch.sigmoid(logits[0, 0]) if self.single_logit else torch.softmax(logits, dim=-1)[0, self.fake_index]
        score.backward()
        if not activations or not gradients:
            return float(score.detach()), []
        activation = activations[0].detach()[0, 1:]
        gradient = gradients[0].detach()[0, 1:]
        weights = gradient.mean(dim=-1, keepdim=True)
        cam = torch.relu((activation * weights).sum(dim=-1))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        grid = int(cam.numel() ** 0.5)
        heatmap = cam.reshape(grid, grid).cpu().numpy()
        regions = []
        for y, x in zip(*np.where((heatmap >= np.quantile(heatmap, 0.85)) & (heatmap > 0))):
            regions.append({
                "x": round(x / grid * 100, 1), "y": round(y / grid * 100, 1),
                "w": round(100 / grid, 1), "h": round(100 / grid, 1),
                "intensity": round(float(heatmap[y, x]), 3),
                "label": "ViT Grad-CAM patch",
            })
        return float(score.detach()), regions[:8]

    def predict(self, filepath: str) -> float:
        return self.predict_with_gradcam(filepath)[0]


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
    """Frame-level adapter for DarkVision's published Keras detector."""

    def __init__(self, model_id: str):
        if tf is None:
            raise RuntimeError("TensorFlow is required for the DarkVision video detector.")
        self.model_id = model_id
        source = model_path(model_id)
        saved_model = os.path.join(source, "p1")
        try:
            if not os.path.isdir(saved_model):
                from huggingface_hub import snapshot_download
                snapshot_download(model_id, allow_patterns=["p1/*"], local_dir=source)
            self.model = tf.saved_model.load(saved_model)
            self.infer = self.model.signatures["serving_default"]
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Could not load video checkpoint {saved_model}: {exc}") from exc
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _face_or_frame(self, frame: Image.Image) -> np.ndarray:
        image = np.asarray(frame.convert("RGB"))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
        if len(faces):
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            image = image[y:y + h, x:x + w]
        return cv2.resize(image, (224, 224)).astype(np.float32) / 255.0

    def predict_frames(self, frames):
        batch = np.stack([self._face_or_frame(frame) for frame in frames])
        predictions = self.infer(input_2=tf.convert_to_tensor(batch))["prediction"].numpy()
        scores = predictions[:, 1] if predictions.shape[-1] > 1 else predictions[:, 0]
        frame_scores = [round(float(score), 4) for score in scores]
        score = float(np.mean(scores))
        regions = [
            {"x": round(index / max(len(scores), 1) * 100, 1), "y": 5,
             "w": round(100 / max(len(scores), 1), 1), "h": 90,
             "intensity": value, "label": "Deepfake frame signal"}
            for index, value in enumerate(frame_scores) if value >= 0.5
        ][:8]
        return score, regions


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
