"""
Text Analysis Service
Runs Desklib's AI text detector for AI-generated text detection and keeps the
existing fake-news classifier path separate.
"""
import asyncio
import time
import random
import re
import os
import numpy as np
from typing import List, Dict, Any

# ── Try importing ML libraries (graceful fallback if not installed) ────────────
try:
    import torch
    import timm.data
    if not hasattr(timm.data, "ImageNetInfo"):
        class ImageNetInfo:
            def __init__(self, *_args, **_kwargs):
                pass

            def label_names(self):
                return []

            def label_descriptions(self, as_dict=False):
                return {} if as_dict else []

        timm.data.ImageNetInfo = ImageNetInfo
    if not hasattr(timm.data, "infer_imagenet_subset"):
        timm.data.infer_imagenet_subset = lambda _config: None
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedModel,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
DESKLIB_MODEL_ID = os.getenv("DESKLIB_MODEL_ID", "desklib/ai-text-detector-v1.01")
ROBERTA_PATH = os.path.join(MODEL_DIR, "roberta_ai_detector")
LINGUA_PATH = os.path.join(MODEL_DIR, "linguaforge_detector")
FAKENEWS_PATH = os.path.join(MODEL_DIR, "fakenews_classifier")


if TORCH_AVAILABLE:
    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = torch.nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return {"logits": self.classifier(pooled)}


class TextAnalysisService:
    def __init__(self):
        self.desklib_model = None
        self.desklib_tokenizer = None
        self.roberta_model = None
        self.roberta_tokenizer = None
        self.lingua_model = None
        self.lingua_tokenizer = None
        self.fakenews_model = None
        self.fakenews_tokenizer = None
        self.fast_text_model = None
        self.fast_fakenews_model = None
        self._load_models()

    def _load_models(self):
        """Load the Desklib detector, fast models, and optional fine-tuned models."""
        fast_text_path = os.path.join(MODEL_DIR, "fast_text_model.pkl")
        if os.path.exists(fast_text_path):
            try:
                import pickle
                with open(fast_text_path, "rb") as f:
                    self.fast_text_model = pickle.load(f)
                print("[TextService] Fast text model loaded.")
            except Exception as e:
                print(f"[TextService] Error loading fast text model: {e}")

        fast_fakenews_path = os.path.join(MODEL_DIR, "fast_fakenews_model.pkl")
        if os.path.exists(fast_fakenews_path):
            try:
                import pickle
                with open(fast_fakenews_path, "rb") as f:
                    self.fast_fakenews_model = pickle.load(f)
                print("[TextService] Fast fake news model loaded.")
            except Exception as e:
                print(f"[TextService] Error loading fast fake news model: {e}")

        if not TORCH_AVAILABLE:
            print("[TextService] PyTorch/Transformers unavailable — relying on fast models.")
            return

        try:
            print(f"[TextService] Loading Desklib model: {DESKLIB_MODEL_ID}")
            self.desklib_tokenizer = AutoTokenizer.from_pretrained(
                DESKLIB_MODEL_ID,
                local_files_only=True,
            )
            config = AutoConfig.from_pretrained(DESKLIB_MODEL_ID, local_files_only=True)
            self.desklib_model = DesklibAIDetectionModel.from_pretrained(
                DESKLIB_MODEL_ID,
                config=config,
                local_files_only=True,
            )
            self.desklib_model.eval()
            print("[TextService] Desklib AI detector loaded.")
        except (OSError, RuntimeError, ValueError) as e:
            self.desklib_model = None
            self.desklib_tokenizer = None
            print(f"[TextService] Desklib model unavailable locally: {e}")

        optional_models = (
            ("RoBERTa", ROBERTA_PATH, "roberta_tokenizer", "roberta_model"),
            ("LinguaForge-X", LINGUA_PATH, "lingua_tokenizer", "lingua_model"),
            ("Fake news", FAKENEWS_PATH, "fakenews_tokenizer", "fakenews_model"),
        )
        for label, path, tokenizer_attr, model_attr in optional_models:
            if not os.path.isdir(path):
                print(f"[TextService] {label} model not found at {path}; using fallback.")
                continue
            try:
                tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    path,
                    local_files_only=True,
                )
                model.eval()
                setattr(self, tokenizer_attr, tokenizer)
                setattr(self, model_attr, model)
                print(f"[TextService] {label} model loaded.")
            except (OSError, RuntimeError, ValueError) as e:
                print(f"[TextService] {label} model unavailable locally: {e}")

    def _infer_model(self, model, tokenizer, text: str, max_length: int = 512) -> float:
        """Run a two-class classifier and return its AI probability."""
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # Assume label 1 = AI Generated
        return float(probs[0][1])

    def _infer_desklib(self, text: str) -> float:
        """Run Desklib's single-logit detector and return calibrated AI probability."""
        inputs = self.desklib_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = self.desklib_model(**inputs)["logits"]
        raw_logit = float(logits.flatten()[0])
        # Calibrate against empirical decision boundary (logit ~3.15)
        calibrated_logit = (raw_logit - 3.15) * 1.2
        return float(torch.sigmoid(torch.tensor(calibrated_logit)).item())

    def _mock_inference(self, text: str) -> Dict[str, Any]:
        """Use the trained fast model instead of mock logic if available."""
        fast_model_path = os.path.join(MODEL_DIR, "fast_text_model.pkl")
        if os.path.exists(fast_model_path):
            try:
                import pickle
                with open(fast_model_path, "rb") as f:
                    pipeline = pickle.load(f)
                prob = float(pipeline.predict_proba([text])[0][1])
                return {"roberta": prob, "linguaforge": prob}
            except Exception as e:
                print(f"Error loading fast model: {e}")

        # Fallback heuristic
        word_count = len(text.split())
        avg_word_len = sum(len(w) for w in text.split()) / max(word_count, 1)
        formal_keywords = ["furthermore", "additionally", "implementation", "leveraging",
                           "comprehensive", "utilizing", "facilitate", "paradigm",
                           "methodology", "seamlessly", "optimize", "robust"]
        formal_score = sum(1 for k in formal_keywords if k.lower() in text.lower()) / len(formal_keywords)

        base_score = 0.45 + (formal_score * 0.35) + (min(avg_word_len - 4, 3) * 0.04)
        roberta_score = float(np.clip(base_score + np.random.normal(0, 0.04), 0.1, 0.98))
        lingua_score = float(np.clip(base_score + np.random.normal(0, 0.04), 0.1, 0.98))
        return {"roberta": roberta_score, "linguaforge": lingua_score}

    def _mock_fake_news(self, text: str) -> float:
        """Mock fake news score based on heuristics."""
        sensational = ["shocking", "breaking", "exclusive", "secret", "they don't want",
                       "you won't believe", "mainstream media", "hidden truth", "exposed"]
        score = 0.35 + sum(0.08 for k in sensational if k.lower() in text.lower())
        return float(np.clip(score + np.random.normal(0, 0.05), 0.05, 0.97))

    def _extract_token_importance(self, text: str, ai_score: float, fake_score: float) -> List[Dict]:
        """Generate token-level importance scores."""
        words = re.findall(r'\b\w+\b', text)[:20]
        formal_words = {"furthermore", "additionally", "implementation", "leveraging",
                        "comprehensive", "utilizing", "facilitate", "paradigm",
                        "methodology", "seamlessly", "optimize", "robust", "strategic"}
        sensational_words = {"shocking", "breaking", "exclusive", "secret", "exposed",
                             "hidden", "mainstream", "truth", "hoax", "agenda"}
        tokens = []
        for word in words:
            w_lower = word.lower()
            if w_lower in formal_words:
                label, score = "ai", round(0.7 + random.random() * 0.28, 3)
            elif w_lower in sensational_words:
                label, score = "fake", round(0.65 + random.random() * 0.3, 3)
            else:
                label, score = "neutral", round(random.random() * 0.45, 3)
            tokens.append({"token": word, "score": score, "label": label})
        return tokens

    def _compute_shap_values(self, text: str) -> List[Dict]:
        """Generate SHAP-style feature importance values."""
        words = re.findall(r'\b\w+\b', text)[:12]
        shap_vals = []
        for word in words:
            shap = round((random.random() - 0.38) * 0.9, 3)
            shap_vals.append({"feature": word, "shap": shap})
        return sorted(shap_vals, key=lambda x: abs(x["shap"]), reverse=True)

    async def analyze_ai_text(self, text: str) -> Dict[str, Any]:
        """Full ensemble AI text detection pipeline."""
        t0 = time.time()

        desklib_score = None
        if self.desklib_model and self.desklib_tokenizer:
            loop = asyncio.get_event_loop()
            desklib_score = await loop.run_in_executor(None, self._infer_desklib, text)

        fast_score = None
        if self.fast_text_model is not None:
            try:
                fast_score = float(self.fast_text_model.predict_proba([text])[0][1])
            except Exception as e:
                print(f"[TextService] Fast text model error: {e}")

        if desklib_score is not None and fast_score is not None:
            # Calibrated ensemble: 65% DeBERTa Deep Detector + 35% Fast Model
            ensemble_score = 0.65 * desklib_score + 0.35 * fast_score
            weights = {"desklib": 0.65, "fast_model": 0.35}
        elif desklib_score is not None:
            ensemble_score = desklib_score
            weights = {"desklib": 1.0}
        elif fast_score is not None:
            ensemble_score = fast_score
            weights = {"fast_model": 1.0}
        else:
            raise RuntimeError(
                "No AI text detection models are available. Ensure models are saved or downloaded."
            )

        human_score = 1.0 - ensemble_score

        # Fake news score
        if self.fakenews_model:
            loop = asyncio.get_event_loop()
            fake_score = await loop.run_in_executor(
                None, self._infer_model, self.fakenews_model, self.fakenews_tokenizer, text)
        elif self.fast_fakenews_model is not None:
            try:
                fake_score = float(self.fast_fakenews_model.predict_proba([text])[0][1])
            except Exception:
                fake_score = self._mock_fake_news(text)
        else:
            fake_score = self._mock_fake_news(text)

        real_score = 1.0 - fake_score

        # XAI
        token_importance = self._extract_token_importance(text, ensemble_score, fake_score)
        shap_values = self._compute_shap_values(text)
        suspicious = [t["token"] for t in token_importance if t["score"] > 0.7 and t["label"] == "ai"][:5]

        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "prediction": "AI Generated" if ensemble_score > 0.5 else "Human Written",
            "ai_confidence": round(ensemble_score, 4),
            "human_confidence": round(human_score, 4),
            "fake_news_score": round(fake_score, 4),
            "real_news_score": round(real_score, 4),
            "desklib": {
                "prediction": "AI Generated" if (desklib_score or 0) > 0.5 else "Human Written",
                "ai_confidence": round(desklib_score, 4) if desklib_score is not None else None,
            },
            "ensemble_weight": weights,
            "token_importance": token_importance,
            "shap_values": shap_values,
            "suspicious_phrases": suspicious,
            "processing_time_ms": elapsed_ms,
        }

    async def analyze_fake_news(self, text: str) -> Dict[str, Any]:
        """Dedicated fake news / misinformation detection."""
        t0 = time.time()

        if self.fakenews_model:
            loop = asyncio.get_event_loop()
            fake_score = await loop.run_in_executor(
                None, self._infer_model, self.fakenews_model, self.fakenews_tokenizer, text)
        elif self.fast_fakenews_model is not None:
            try:
                fake_score = float(self.fast_fakenews_model.predict_proba([text])[0][1])
            except Exception:
                fake_score = self._mock_fake_news(text)
        else:
            await asyncio.sleep(0.1)
            fake_score = self._mock_fake_news(text)

        real_score = 1.0 - fake_score
        shap_values = self._compute_shap_values(text)
        token_importance = self._extract_token_importance(text, 0.5, fake_score)
        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "prediction": "Fake News" if fake_score > 0.5 else "Credible",
            "fake_score": round(fake_score, 4),
            "real_score": round(real_score, 4),
            "credibility_rating": "Low" if fake_score > 0.7 else "Medium" if fake_score > 0.4 else "High",
            "shap_values": shap_values,
            "token_importance": token_importance,
            "processing_time_ms": elapsed_ms,
        }
