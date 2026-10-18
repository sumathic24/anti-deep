"""
Text Analysis Service
Detects AI-generated text. Model priority:
  1. a pre-trained Hugging Face detector, once downloaded (see services/pretrained.py),
  2. the RoBERTa + LinguaForge-X ensemble, when trained models are in models/saved/.
With neither, text analysis is unavailable. Explanations are real SHAP values computed
on whichever model is active, so the same text always gets the same result.
"""
import asyncio
import json
import os
import pickle
import re
import threading
import time
import numpy as np
import shap
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import Any, Dict, List

# ── Try importing ML libraries (transformer models are optional) ──────────────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from services.linguaforge import load_linguaforge, extract_linguistic_features, WEIGHTS_FILE
    from services.pretrained import TextDetector, TEXT_MODEL_ID, load_detector
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
ROBERTA_PATH = os.path.join(MODEL_DIR, "roberta_ai_detector")
LINGUA_PATH = os.path.join(MODEL_DIR, "linguaforge_detector")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "ensemble")

MAX_LENGTH = 512
BATCH_SIZE = 16
MAX_DISPLAY_TOKENS = 150
TOP_SHAP_FEATURES = 12


class ModelUnavailableError(RuntimeError):
    pass


class TextAnalysisService:
    def __init__(self):
        self.pretrained = None     # TextDetector (Hugging Face)
        self.roberta = None        # (model, tokenizer)
        self.lingua = None         # (model, tokenizer)
        self.weights = {"roberta": 0.55, "linguaforge": 0.45}
        self.meta_learner = None
        self._lock = threading.Lock()
        self._load_models()

        self.explainer = None
        if self.available:
            # Transformers are slow on CPU, so SHAP gets a small evaluation budget.
            self.shap_max_evals = int(os.getenv("SHAP_MAX_EVALS", 120))
            self.explainer = shap.Explainer(self._predict_ai_proba, shap.maskers.Text(r"\W+"))
            # The first SHAP call JIT-compiles internals; do it now instead of on a user request.
            self._explain("Warm-up sentence for the explainer so the first request is fast.")
        print(f"[TextService] Active model: {self.model_used}")

    # ── Loading ───────────────────────────────────────────────────────────────
    def _load_models(self):
        if TORCH_AVAILABLE:
            self.pretrained = load_detector(TextDetector, TEXT_MODEL_ID, "TextService")
            if self.pretrained:
                return

            if os.path.isdir(ROBERTA_PATH):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH)
                    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_PATH).eval()
                    self.roberta = (model, tokenizer)
                    print("[TextService] RoBERTa model loaded.")
                except Exception as e:
                    print(f"[TextService] RoBERTa load error: {e}")

            if os.path.exists(os.path.join(LINGUA_PATH, WEIGHTS_FILE)):
                try:
                    self.lingua = load_linguaforge(LINGUA_PATH)
                    print("[TextService] LinguaForge-X model loaded.")
                except Exception as e:
                    print(f"[TextService] LinguaForge-X load error: {e}")

            if self.roberta and self.lingua:
                self._load_ensemble_config()
        else:
            print("[TextService] PyTorch not available — transformer models disabled.")

    def _load_ensemble_config(self):
        """Use the weights / meta-learner chosen by training/train_ensemble.py, if present."""
        config_path = os.path.join(ENSEMBLE_PATH, "ensemble_config.json")
        if not os.path.exists(config_path):
            return
        try:
            with open(config_path) as f:
                config = json.load(f)
            self.weights = {"roberta": float(config["roberta_weight"]),
                            "linguaforge": float(config["linguaforge_weight"])}
            meta_path = os.path.join(ENSEMBLE_PATH, "meta_learner.pkl")
            if config.get("strategy") == "meta_learner" and os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.meta_learner = pickle.load(f)
            print(f"[TextService] Ensemble config loaded ({config.get('strategy')}).")
        except Exception as e:
            print(f"[TextService] Ensemble config error: {e} — using default weights.")

    @property
    def available(self) -> bool:
        return bool(self.pretrained or self.roberta or self.lingua)

    @property
    def model_used(self) -> str:
        if self.pretrained:
            return f"{self.pretrained.model_id} (pre-trained)"
        if self.roberta and self.lingua:
            return "RoBERTa + LinguaForge-X ensemble"
        if self.roberta:
            return "RoBERTa"
        if self.lingua:
            return "LinguaForge-X"
        return "No model available"

    # ── Inference ─────────────────────────────────────────────────────────────
    def _roberta_proba(self, texts: List[str]) -> np.ndarray:
        model, tokenizer = self.roberta
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            enc = tokenizer(texts[i:i + BATCH_SIZE], return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH, padding=True)
            with torch.no_grad():
                logits = model(**enc).logits
            out.append(torch.softmax(logits, dim=-1)[:, 1].numpy())  # label 1 = AI Generated
        return np.concatenate(out)

    def _lingua_proba(self, texts: List[str]) -> np.ndarray:
        model, tokenizer = self.lingua
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH, padding=True)
            ling = torch.tensor(np.stack([extract_linguistic_features(t) for t in batch]))
            with torch.no_grad():
                _, logits = model(enc["input_ids"], enc["attention_mask"], ling)
            out.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
        return np.concatenate(out)

    def _model_probas(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """AI probability from each active model."""
        if self.pretrained:
            return {"pretrained": self.pretrained.predict(texts)}
        probas = {}
        if self.roberta:
            probas["roberta"] = self._roberta_proba(texts)
        if self.lingua:
            probas["linguaforge"] = self._lingua_proba(texts)
        return probas

    def _combine(self, probas: Dict[str, np.ndarray]) -> np.ndarray:
        if "roberta" in probas and "linguaforge" in probas:
            r, l = probas["roberta"], probas["linguaforge"]
            if self.meta_learner is not None:
                return self.meta_learner.predict_proba(np.column_stack([r, l]))[:, 1]
            w_r, w_l = self.weights["roberta"], self.weights["linguaforge"]
            return (r * w_r + l * w_l) / (w_r + w_l)
        return next(iter(probas.values()))

    def _predict_ai_proba(self, texts) -> np.ndarray:
        """Final AI probability for a batch of texts (also the function SHAP explains)."""
        return self._combine(self._model_probas([str(t) for t in texts]))

    # ── Explainability ────────────────────────────────────────────────────────
    def _explain(self, text: str):
        sv = self.explainer([text], max_evals=self.shap_max_evals, silent=True)
        return sv.data[0], sv.values[0]

    @staticmethod
    def _build_explanations(tokens, values, ai_score: float) -> Dict[str, Any]:
        words, contribs = [], []
        for tok, val in zip(tokens, values):
            word = re.sub(r"\W+", "", str(tok))
            if word:
                words.append(word)
                contribs.append(float(val))

        max_abs = max((abs(v) for v in contribs), default=0.0) or 1.0
        token_importance = []
        for word, val in list(zip(words, contribs))[:MAX_DISPLAY_TOKENS]:
            score = abs(val) / max_abs
            label = "neutral" if score < 0.1 else ("ai" if val > 0 else "human")
            token_importance.append({"token": word, "shap": round(val, 4),
                                     "score": round(score, 3), "label": label})

        # SHAP values are additive, so repeated words are summed into one feature.
        # Common filler words ("the", "of", …) are left out of the ranked lists so they
        # don't crowd out meaningful words; they still appear in token_importance.
        totals: Dict[str, float] = {}
        for word, val in zip(words, contribs):
            key = word.lower()
            if len(key) > 1 and key not in ENGLISH_STOP_WORDS:
                totals[key] = totals.get(key, 0.0) + val
        # Words with no measurable effect (< 0.05 percentage points) are dropped from the ranking.
        shap_values = sorted(({"feature": w, "shap": round(v, 4)} for w, v in totals.items() if abs(v) >= 5e-4),
                             key=lambda x: abs(x["shap"]), reverse=True)[:TOP_SHAP_FEATURES]

        suspicious = []
        if ai_score > 0.5:
            suspicious = [f["feature"] for f in shap_values if f["shap"] > 0][:5]

        return {"token_importance": token_importance, "shap_values": shap_values,
                "suspicious_phrases": suspicious}

    # ── Public API ────────────────────────────────────────────────────────────
    def _analyze_sync(self, text: str, explain: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        with self._lock:
            probas = {k: float(v[0]) for k, v in self._model_probas([text]).items()}
            ai_score = float(self._combine({k: np.array([v]) for k, v in probas.items()})[0])
            # SHAP is the slow part on transformer models, so it can be skipped.
            tokens, values = self._explain(text) if explain else ([], [])

        def model_entry(key):
            if key not in probas:
                return None
            return {"prediction": "AI Generated" if probas[key] > 0.5 else "Human Written",
                    "ai_confidence": round(probas[key], 4)}

        is_ensemble = "roberta" in probas and "linguaforge" in probas
        return {
            "prediction": "AI Generated" if ai_score > 0.5 else "Human Written",
            "ai_confidence": round(ai_score, 4),
            "human_confidence": round(1.0 - ai_score, 4),
            "model_used": self.model_used,
            "roberta": model_entry("roberta"),
            "linguaforge": model_entry("linguaforge"),
            "ensemble_strategy": ("meta_learner" if self.meta_learner is not None else "weighted_average") if is_ensemble else None,
            "ensemble_weight": self.weights if is_ensemble and self.meta_learner is None else None,
            "explained": explain,
            **self._build_explanations(tokens, values, ai_score),
            "processing_time_ms": int((time.time() - t0) * 1000),
        }

    async def analyze_ai_text(self, text: str, explain: bool = True) -> Dict[str, Any]:
        """AI-generated text detection, with SHAP explanations unless explain=False."""
        if not self.available:
            raise ModelUnavailableError(
                "No AI text detection model is available on the server. Run "
                "`python download_models.py` in the backend folder, or train a model (see README).")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._analyze_sync, text, explain)
