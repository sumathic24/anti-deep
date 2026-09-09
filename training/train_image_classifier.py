"""
train_image_classifier.py — EfficientNet-B4 + ViT-B/16 for AI Image Detection
Dataset: CIFAKE + DiffusionDB + AI Generated Images Dataset
Label 0: Real Image | Label 1: AI Generated
Usage: python train_image_classifier.py --data_dir ./data/images --model efficientnet
"""
import argparse, os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LABEL_NAMES = ["Real Image", "AI Generated"]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_model(name: str) -> nn.Module:
    if name == "efficientnet":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif name == "vit":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds_all.extend(out.argmax(-1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return total_loss / len(loader), accuracy_score(labels_all, preds_all)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            total_loss += criterion(out, labels).item()
            preds_all.extend(out.argmax(-1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="weighted")
    return total_loss / len(loader), acc, f1, preds_all, labels_all


def plot_cm(labels, preds, model_name, out_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Image Trainer] Model: {args.model} | Device: {device}")

    # Datasets (ImageFolder structure: data/images/train/real/ data/images/train/ai_generated/)
    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=TRAIN_TRANSFORM)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=VAL_TRANSFORM)
    test_ds  = datasets.ImageFolder(os.path.join(args.data_dir, "test"),  transform=VAL_TRANSFORM)

    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_ld  = DataLoader(test_ds,  args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0
    print(f"\nTraining {args.epochs} epochs on {len(train_ds)} images...\n")

    for epoch in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, train_ld, optimizer, criterion, device)
        vl, va, vf1, _, _ = eval_epoch(model, val_ld, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | Train {tl:.4f}/{ta:.4f} | Val {vl:.4f}/{va:.4f} F1={vf1:.4f}")

        if vf1 > best_f1:
            best_f1 = vf1
            save_path = os.path.join(args.output_dir, f"{args.model}_detector.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved (F1={best_f1:.4f})")

    # Test
    _, ta, tf1, tp, tl2 = eval_epoch(model, test_ld, criterion, device)
    print(f"\n[Test] Accuracy: {ta:.4f} | F1: {tf1:.4f}")
    print(classification_report(tl2, tp, target_names=LABEL_NAMES))
    plot_cm(tl2, tp, args.model, args.output_dir)

    metrics = {"model": args.model, "test_accuracy": round(ta, 4),
               "test_f1": round(tf1, 4), "best_val_f1": round(best_f1, 4)}
    with open(os.path.join(args.output_dir, f"{args.model}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Done] {args.model} saved to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="./data/images")
    p.add_argument("--output_dir", default="../backend/models/saved")
    p.add_argument("--model",      default="efficientnet", choices=["efficientnet", "vit"])
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    main(p.parse_args())
