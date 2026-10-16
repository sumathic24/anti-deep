"""
Download the pre-trained detectors the backend uses into the local Hugging Face cache.
Nothing is downloaded until you run this; restart the backend afterwards.

    python download_models.py              # text + image detector
    python download_models.py --text       # text detector only
    python download_models.py --image      # image detector only
    python download_models.py --dry-run    # show what would be downloaded, and its size

Choose other models with the TEXT_DETECTOR_MODEL / IMAGE_DETECTOR_MODEL environment
variables (see README). To free the space again, delete the model's folder from the
cache directory printed at the end.
"""
import argparse
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from services.pretrained import TEXT_MODEL_ID, IMAGE_MODEL_ID

CONFIG_PATTERNS = ["config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "spm.model"]


def files_to_download(model_id):
    """Config + tokenizer files and one copy of the weights (skips legacy/ONNX/duplicate exports)."""
    info = HfApi().model_info(model_id, files_metadata=True)
    sizes = {f.rfilename: f.size or 0 for f in info.siblings}
    weights = "model.safetensors" if "model.safetensors" in sizes else "pytorch_model.bin"
    wanted = [f for f in sizes if f in CONFIG_PATTERNS] + [weights]
    return wanted, sum(sizes.get(f, 0) for f in wanted)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", action="store_true", help="download the text detector only")
    parser.add_argument("--image", action="store_true", help="download the image detector only")
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes without downloading")
    args = parser.parse_args()
    both = not (args.text or args.image)

    targets = []
    if (both or args.text) and TEXT_MODEL_ID:
        targets.append(("Text detector", TEXT_MODEL_ID))
    if (both or args.image) and IMAGE_MODEL_ID:
        targets.append(("Image detector", IMAGE_MODEL_ID))
    if not targets:
        print("Nothing to download (the selected detectors are disabled).")
        return

    plan = [(label, model_id, *files_to_download(model_id)) for label, model_id in targets]
    for label, model_id, files, size in plan:
        print(f"{label}: {model_id}  ({size / 1e6:.0f} MB)\n    files: {', '.join(files)}")
    print(f"Total: {sum(p[3] for p in plan) / 1e6:.0f} MB")
    if args.dry_run:
        return

    for label, model_id, files, _ in plan:
        print(f"\nDownloading {model_id} …")
        snapshot_download(model_id, allow_patterns=files)
    print(f"\nDone. Models are stored in {HF_HUB_CACHE}\nRestart the backend to use them.")


if __name__ == "__main__":
    main()
