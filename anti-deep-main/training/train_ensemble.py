"""
train_ensemble.py — RoBERTa + LinguaForge-X Ensemble Training
Loads both trained models, runs inference on validation/test set,
learns optimal weights via grid search or logistic regression meta-learner.
Usage: python train_ensemble.py --roberta_dir ... --lingua_dir ... --data_path ...
"""
import argparse, os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from train_linguaforge import LinguaForgeX, extract_linguistic_features

LABEL_NAMES = ["Human Written", "AI Generated"]


class EnsembleDataset(Dataset):
    def __init__(self, filepath, roberta_tok, lingua_tok, max_len=512):
        self.data = [json.loads(l) for l in open(filepath, encoding="utf-8")]
        self.roberta_tok = roberta_tok
        self.lingua_tok  = lingua_tok
        self.max_len     = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        re = self.roberta_tok(text, max_length=self.max_len, truncation=True,
                              padding="max_length", return_tensors="pt")
        le = self.lingua_tok(text, max_length=self.max_len, truncation=True,
                             padding="max_length", return_tensors="pt")
        return {
            "r_ids":  re["input_ids"].squeeze(0),
            "r_mask": re["attention_mask"].squeeze(0),
            "l_ids":  le["input_ids"].squeeze(0),
            "l_mask": le["attention_mask"].squeeze(0),
            "ling":   torch.tensor(extract_linguistic_features(text)),
            "label":  torch.tensor(int(item["label"]), dtype=torch.long),
        }


def get_probabilities(roberta, lingua_model, loader, device):
    roberta.eval(); lingua_model.eval()
    r_probs, l_probs, all_labels = [], [], []
    with torch.no_grad():
        for b in loader:
            # RoBERTa
            r_out = roberta(input_ids=b["r_ids"].to(device),
                            attention_mask=b["r_mask"].to(device))
            r_p = torch.softmax(r_out.logits, dim=-1)[:, 1].cpu().numpy()
            # LinguaForge
            _, logits = lingua_model(b["l_ids"].to(device), b["l_mask"].to(device),
                                     b["ling"].to(device))
            l_p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            r_probs.extend(r_p)
            l_probs.extend(l_p)
            all_labels.extend(b["label"].numpy())
    return np.array(r_probs), np.array(l_probs), np.array(all_labels)


def grid_search_weights(r_probs, l_probs, labels):
    """Find best weighted average via grid search."""
    best_f1, best_w = 0, 0.5
    for w in np.arange(0.1, 1.0, 0.05):
        ensemble = w * r_probs + (1 - w) * l_probs
        preds = (ensemble > 0.5).astype(int)
        f1 = f1_score(labels, preds, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_w = round(w, 2)
    return best_w, round(1 - best_w, 2), best_f1


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Ensemble] Device: {device}")

    # Load models
    roberta_tok = AutoTokenizer.from_pretrained(args.roberta_dir)
    roberta     = AutoModelForSequenceClassification.from_pretrained(args.roberta_dir).to(device)
    lingua_tok  = AutoTokenizer.from_pretrained(args.lingua_dir)
    lingua      = LinguaForgeX().to(device)
    lingua.load_state_dict(torch.load(
        os.path.join(args.lingua_dir, "linguaforge_model.pth"), map_location=device))

    dataset = EnsembleDataset(args.data_path, roberta_tok, lingua_tok)
    split = int(len(dataset) * 0.80)
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    print("Extracting probabilities from both models...")
    r_probs, l_probs, labels = get_probabilities(roberta, lingua, val_ld, device)

    # Grid search for optimal weights
    r_weight, l_weight, grid_f1 = grid_search_weights(r_probs, l_probs, labels)
    print(f"\n[Grid Search] Best weights → RoBERTa={r_weight}, LinguaForge={l_weight}, F1={grid_f1:.4f}")

    # Meta-learner: Logistic Regression on stacked probabilities
    X = np.column_stack([r_probs, l_probs])
    meta = LogisticRegression(max_iter=500)
    meta.fit(X, labels)
    meta_preds = meta.predict(X)
    meta_f1 = f1_score(labels, meta_preds, average="weighted")
    print(f"[Meta-Learner] Logistic Regression F1: {meta_f1:.4f}")
    print(classification_report(labels, meta_preds, target_names=LABEL_NAMES))

    # Choose best strategy
    if meta_f1 >= grid_f1:
        strategy = "meta_learner"
        final_f1 = meta_f1
        import pickle
        with open(os.path.join(args.output_dir, "meta_learner.pkl"), "wb") as f:
            pickle.dump(meta, f)
        print("✓ Using Meta-Learner strategy")
    else:
        strategy = "weighted_average"
        final_f1 = grid_f1
        print(f"✓ Using Weighted Average strategy (RoBERTa={r_weight}, Lingua={l_weight})")

    config = {
        "strategy": strategy,
        "roberta_weight": r_weight,
        "linguaforge_weight": l_weight,
        "meta_learner_f1": round(meta_f1, 4),
        "grid_search_f1": round(grid_f1, 4),
        "final_f1": round(final_f1, 4),
    }
    with open(os.path.join(args.output_dir, "ensemble_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[Done] Ensemble config saved. Final F1: {final_f1:.4f}")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roberta_dir", default="../backend/models/saved/roberta_ai_detector")
    p.add_argument("--lingua_dir",  default="../backend/models/saved/linguaforge_detector")
    p.add_argument("--data_path",   default="./data/text/ai_human.jsonl")
    p.add_argument("--output_dir",  default="../backend/models/saved/ensemble")
    p.add_argument("--batch_size",  type=int, default=32)
    main(p.parse_args())
