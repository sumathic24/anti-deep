"""
LinguaForge-X model definition for inference.
Must stay in sync with training/train_linguaforge.py — the saved state_dict
only loads into a module with exactly the same layers and feature extractor.
"""
import os
import re
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

BERT_NAME = "bert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = 512
WEIGHTS_FILE = "linguaforge_model.pth"

FORMAL_WORDS = {"furthermore","additionally","implementation","leveraging","comprehensive",
                "utilizing","facilitate","paradigm","methodology","seamlessly","optimize",
                "robust","strategic","innovative","synergy","holistic","streamline"}


def extract_linguistic_features(text: str) -> np.ndarray:
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    avg_sent_len = len(words) / max(len(sentences), 1)
    punct_density = len(re.findall(r'[,;:—–]', text)) / max(len(words), 1)
    formal_ratio = sum(1 for w in words if w in FORMAL_WORDS) / max(len(words), 1)
    ttr = len(set(words)) / max(len(words), 1)
    avg_syllables = np.mean([max(1, len(re.findall(r'[aeiou]', w))) for w in words]) if words else 0
    passive = 1.0 if re.search(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', text) else 0.0
    sent_count = float(len(sentences))
    feats = np.array([avg_word_len, sent_count, avg_sent_len, punct_density,
                      formal_ratio, ttr, avg_syllables, passive], dtype=np.float32)
    feats = np.nan_to_num(feats)
    return np.clip(feats / (feats.max() + 1e-8), 0, 1)


class LinguaForgeX(nn.Module):
    """BERT CLS embedding (768) + linguistic features (8) → MLP classifier."""
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.ling_proj = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Dropout(0.1))
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_LABELS),
        )

    def forward(self, input_ids, attention_mask, ling, labels=None):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        fused = torch.cat([cls, self.ling_proj(ling)], dim=-1)
        logits = self.classifier(fused)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return loss, logits


def load_linguaforge(model_dir: str):
    """Load a trained LinguaForge-X from model_dir. Returns (model, tokenizer)."""
    # Newer training runs save the BERT config next to the weights; older ones
    # only saved the tokenizer, so fall back to the base model's config.
    config_source = model_dir if os.path.exists(os.path.join(model_dir, "config.json")) else BERT_NAME
    bert = AutoModel.from_config(AutoConfig.from_pretrained(config_source))
    model = LinguaForgeX(bert)

    state = torch.load(os.path.join(model_dir, WEIGHTS_FILE), map_location="cpu", weights_only=True)
    # position_ids is a buffer that some transformers versions save and others don't.
    state.pop("bert.embeddings.position_ids", None)
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer
