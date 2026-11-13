"""
train_linguaforge.py — LinguaForge-X: BERT + linguistic feature fusion model.
Label 0: Human Written | Label 1: AI Generated
Usage: python train_linguaforge.py --data_path ./data/text/ai_human.jsonl
"""
import argparse, os, json, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

BERT_NAME = "bert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = 512
LABEL_NAMES = ["Human Written", "AI Generated"]

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
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BERT_NAME)
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


class LinguaDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.data = [json.loads(l) for l in open(filepath, encoding="utf-8")]
        self.tokenizer = tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(item["text"], max_length=MAX_LENGTH, truncation=True,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "linguistic_features": torch.tensor(extract_linguistic_features(item["text"])),
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
        }


def run_epoch(model, loader, optimizer=None, scheduler=None, device="cpu", train=True):
    model.train() if train else model.eval()
    preds_all, labels_all, total_loss = [], [], 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            ling = batch["linguistic_features"].to(device)
            labels = batch["label"].to(device)
            if train: optimizer.zero_grad()
            loss, logits = model(ids, mask, ling, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
            total_loss += loss.item()
            preds_all.extend(logits.argmax(-1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="weighted")
    return total_loss / len(loader), acc, f1, preds_all, labels_all


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LinguaForge-X] Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    model = LinguaForgeX().to(device)
    dataset = LinguaDataset(args.data_path, tokenizer)
    v, te = int(len(dataset)*0.15), int(len(dataset)*0.10)
    tr = len(dataset) - v - te
    train_ds, val_ds, test_ds = random_split(dataset, [tr, v, te])
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=2)
    val_ld   = DataLoader(val_ds,   args.batch_size, num_workers=2)
    test_ld  = DataLoader(test_ds,  args.batch_size, num_workers=2)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_ld) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, steps//10, steps)
    best_f1 = 0
    print(f"Training {args.epochs} epochs on {tr} samples...\n")
    for ep in range(1, args.epochs+1):
        tl, ta, _ ,_,_ = run_epoch(model, train_ld, optimizer, scheduler, device, train=True)
        vl, va, vf1, _,_ = run_epoch(model, val_ld, device=device, train=False)
        print(f"Epoch {ep} | Train {tl:.4f}/{ta:.4f} | Val {vl:.4f}/{va:.4f} F1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "linguaforge_model.pth"))
            tokenizer.save_pretrained(args.output_dir)
            print(f"  ✓ Saved (F1={best_f1:.4f})")
    _, ta, tf1, tp, tl2 = run_epoch(model, test_ld, device=device, train=False)
    print(f"\n[Test] Acc: {ta:.4f}  F1: {tf1:.4f}")
    print(classification_report(tl2, tp, target_names=LABEL_NAMES))
    json.dump({"test_accuracy": round(ta,4), "test_f1": round(tf1,4)},
              open(os.path.join(args.output_dir,"linguaforge_metrics.json"),"w"), indent=2)
    print(f"[Done] Saved to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  default="./data/text/ai_human.jsonl")
    p.add_argument("--output_dir", default="../backend/models/saved/linguaforge_detector")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=2e-5)
    main(p.parse_args())
