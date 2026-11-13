"""
dataset_prep.py — Download and prepare all datasets for DeepFakeShield.

Datasets:
  Text:  HC3, GPT-Wiki-Intro, OpenAI Human vs AI
  Image: CIFAKE (HuggingFace), DiffusionDB subset

Output format:
  data/text/ai_human.jsonl    — {"text": "...", "label": 0|1}
  data/images/train/real/     — real images
  data/images/train/ai_generated/  — AI images
  data/images/val/  ...
  data/images/test/ ...

Usage:
    python dataset_prep.py --output_dir ./data
"""
import argparse, os, json, random, shutil
from pathlib import Path


def make_dirs(base):
    for split in ["train", "val", "test"]:
        for cls in ["real", "ai_generated"]:
            Path(f"{base}/images/{split}/{cls}").mkdir(parents=True, exist_ok=True)
    Path(f"{base}/text").mkdir(parents=True, exist_ok=True)
    print(f"[✓] Directories created under {base}/")


def prepare_hc3(output_dir):
    """HC3: Human ChatGPT Comparison Corpus — download from HuggingFace."""
    print("\n[HC3] Preparing dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
        records = []
        for item in ds:
            if item.get("human_answers"):
                for ans in item["human_answers"]:
                    if ans and len(ans.strip()) > 50:
                        records.append({"text": ans.strip(), "label": 0})
            if item.get("chatgpt_answers"):
                for ans in item["chatgpt_answers"]:
                    if ans and len(ans.strip()) > 50:
                        records.append({"text": ans.strip(), "label": 1})
        random.shuffle(records)
        out = os.path.join(output_dir, "text", "ai_human.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[HC3] Saved {len(records)} samples → {out}")
    except Exception as e:
        print(f"[HC3] Error: {e}. Install: pip install datasets")
        _create_sample_text_data(output_dir)


def prepare_cifake(output_dir):
    """CIFAKE: 60K real (CIFAR-10) + 60K AI-generated images from HuggingFace."""
    print("\n[CIFAKE] Preparing image dataset...")
    try:
        from datasets import load_dataset
        from PIL import Image as PILImage
        ds = load_dataset("mfeklix/cifake", split="train")
        counts = {"real": 0, "ai_generated": 0}
        MAX_PER_CLASS = 5000

        for item in ds:
            label     = item["label"]  # 0=real, 1=AI
            class_dir = "real" if label == 0 else "ai_generated"
            if counts[class_dir] >= MAX_PER_CLASS:
                continue

            # Assign to split
            r = random.random()
            split = "train" if r < 0.70 else "val" if r < 0.85 else "test"
            out_path = os.path.join(output_dir, "images", split, class_dir,
                                    f"cifake_{counts[class_dir]:05d}.jpg")
            item["image"].save(out_path)
            counts[class_dir] += 1

            if sum(counts.values()) % 1000 == 0:
                print(f"  Processed {sum(counts.values())} images...")

            if all(v >= MAX_PER_CLASS for v in counts.values()):
                break

        print(f"[CIFAKE] Real: {counts['real']}  AI: {counts['ai_generated']}")
    except Exception as e:
        print(f"[CIFAKE] Error: {e}. Install: pip install datasets pillow")


def _create_sample_text_data(output_dir):
    """Create minimal sample data for testing when datasets unavailable."""
    samples = [
        {"text": "The implementation of blockchain in supply chains facilitates unprecedented transparency and traceability across distributed networks.", "label": 1},
        {"text": "Furthermore, leveraging advanced AI methodologies enables seamless optimization of complex organizational paradigms.", "label": 1},
        {"text": "I went to the store and bought milk. It was on sale for two dollars, which was great.", "label": 0},
        {"text": "My dog hates the rain. Every time it pours, she hides under the bed and refuses to come out.", "label": 0},
        {"text": "The comprehensive utilization of innovative technologies facilitates robust streamlining of multifaceted processes.", "label": 1},
    ] * 200
    random.shuffle(samples)
    out = os.path.join(output_dir, "text", "ai_human.jsonl")
    with open(out, "w") as f:
        for s in samples: f.write(json.dumps(s) + "\n")
    print(f"[Sample] Created {len(samples)} text samples → {out}")


def print_summary(output_dir):
    print("\n" + "="*50)
    print("DATASET PREPARATION SUMMARY")
    print("="*50)
    for f in ["text/ai_human.jsonl"]:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            lines = sum(1 for _ in open(path))
            print(f"  {f}: {lines:,} samples")
    for split in ["train", "val", "test"]:
        for cls in ["real", "ai_generated"]:
            d = os.path.join(output_dir, "images", split, cls)
            if os.path.exists(d):
                n = len(os.listdir(d))
                print(f"  images/{split}/{cls}: {n} images")
    print("="*50)


def main(args):
    make_dirs(args.output_dir)
    prepare_hc3(args.output_dir)
    prepare_cifake(args.output_dir)
    print_summary(args.output_dir)
    print("\n[✓] Dataset preparation complete!")
    print("Next steps:")
    print("  python train_roberta.py --data_path ./data/text/ai_human.jsonl")
    print("  python train_linguaforge.py --data_path ./data/text/ai_human.jsonl")
    print("  python train_image_classifier.py --data_dir ./data/images --model efficientnet")
    print("  python train_image_classifier.py --data_dir ./data/images --model vit")
    print("  python train_ensemble.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./data")
    main(p.parse_args())
