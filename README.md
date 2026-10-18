# DeepFakeShield 🛡️

**Automated Detection of Deepfakes and Misinformation in Social Media**

> Final Year Major Project — B.Tech CSE (Data Science)
> Aligned with **SDG 16: Peace, Justice and Strong Institutions**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

DeepFakeShield is a production-quality multimodal AI detection platform that combines:

- **NLP (RoBERTa + LinguaForge-X)** — AI-generated text detection
- **Computer Vision (EfficientNet + ViT)** — AI-generated image detection
- **Explainable AI (SHAP, GradCAM)** — Full reasoning transparency
- **Ensemble Learning** — Weighted prediction fusion for maximum accuracy

Every prediction includes **full explainability on the same dashboard** — no separate pages.

---

## 🏗️ Project Structure

```
anti-deep/
├── frontend/                    # React + Vite + Tailwind + Framer Motion
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx      # Hero, features, stats, pipeline, CTA
│   │   │   ├── Dashboard.jsx    # Analytics, charts, recent analyses
│   │   │   ├── TextAnalysis.jsx # Input → verdict + XAI + SHAP + ensemble
│   │   │   ├── ImageAnalysis.jsx# Upload → verdict + GradCAM + model compare
│   │   │   ├── About.jsx        # Models, datasets, tech stack, SDG
│   │   │   └── NotFound.jsx     # 404 page
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── Logo.jsx
│   │   │   ├── Footer.jsx       # incl. live API status
│   │   │   └── ui.jsx           # Shared UI (verdict card, alerts, collapsibles)
│   │   └── utils/api.js         # Axios API calls
│   └── package.json
│
├── backend/                     # FastAPI + PyTorch
│   ├── main.py                  # App entry + CORS
│   ├── routers/
│   │   ├── text.py              # /api/text/analyze
│   │   ├── image.py             # /api/image/analyze
│   │   └── stats.py             # /api/stats (from recorded history)
│   ├── services/
│   │   ├── text_service.py      # RoBERTa + LinguaForge-X ensemble + SHAP
│   │   ├── linguaforge.py       # LinguaForge-X model definition
│   │   ├── image_service.py     # EfficientNet + ViT + GradCAM
│   │   ├── pretrained.py        # Pre-trained Hugging Face detectors (text + image)
│   │   └── history.py           # SQLite analysis history (data/history.db)
│   ├── models/saved/            # Trained model weights (after training)
│   ├── download_models.py       # Downloads the pre-trained detectors
│   └── requirements.txt
│
├── training/                    # Model training scripts
│   ├── dataset_prep.py          # Download + prepare all datasets
│   ├── train_roberta.py         # Fine-tune RoBERTa for AI text detection
│   ├── train_linguaforge.py     # Train LinguaForge-X (BERT + linguistic features)
│   ├── train_ensemble.py        # Grid search + meta-learner ensemble fusion
│   └── train_image_classifier.py# EfficientNet / ViT image classifier
│
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Node.js ≥ 18
- Python ≥ 3.10
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/sumathic24/anti-deep.git
cd anti-deep
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:5173**

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt

# Start FastAPI server
uvicorn main:app --reload --port 8000
```

Backend API at: **http://localhost:8000**
API Docs at: **http://localhost:8000/docs**

> **Note:** No models are bundled with the repo. Download the pre-trained detectors
> (next section) or train your own (see Training section). Without a model:
> - **Text:** analysis is unavailable; the API returns an error explaining how to get a model.
> - **Images:** a metadata heuristic (EXIF + image size). GradCAM needs the trained EfficientNet.
>
> Every result says which model produced it.

---

## ⬇️ Pre-trained Detectors (recommended)

Instead of training from scratch, the backend can use detectors that are already trained and
published on Hugging Face. They are **not** downloaded automatically — run this once:

```bash
cd backend
venv\Scripts\activate                 # Windows  (source venv/bin/activate on Linux/Mac)
python download_models.py --dry-run   # shows what will be downloaded (~590 MB in total)
python download_models.py             # downloads the text + image detectors
```

Restart the backend; its log then shows `Pre-trained detector loaded: …`, and every result
names the model that produced it. Use `--text` or `--image` to download only one of them.

| Task | Default model | Download | License | Published results (by the authors) |
|---|---|---|---|
| AI text | [`Oxidane/tmr-ai-text-detector`](https://huggingface.co/Oxidane/tmr-ai-text-detector) — RoBERTa-base trained on the RAID benchmark | 503 MB | MIT | RAID leaderboard: AUROC 99.28% across all settings incl. adversarial attacks; 97.4% accuracy on a held-out RAID split |
| AI image | [`buildborderless/CommunityForensics-DeepfakeDet-ViT`](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) — ViT-Small trained on 2.7M images from 4,803 generators ([Community Forensics paper](https://arxiv.org/abs/2411.04125)) | 87 MB | MIT | See the paper |

**Other supported models** — set the environment variable, then download:

| Model | Notes |
|---|---|---|---|
| `desklib/ai-text-detector-v1.01` | DeBERTa-v3-large; listed at the top of the RAID leaderboard on its model card. 1.7 GB and several times slower on CPU, so SHAP explanations take much longer. |
| `fakespot-ai/roberta-base-ai-text-detection-v1` | RoBERTa-base by Mozilla Fakespot, Apache-2.0. |
| `Hello-SimpleAI/chatgpt-detector-roberta` | Trained on HC3 (ChatGPT-3.5 answers, 2023) — weaker on newer AI models. |
| `haywoodsloan/ai-image-detector-deploy` | SwinV2 image detector, Apache-2.0, regularly updated. |
| `Organika/sdxl-detector` | Swin image detector, strong on SDXL images; CC BY-NC (non-commercial only). |

```bash
set TEXT_DETECTOR_MODEL=desklib/ai-text-detector-v1.01      # Windows (export ... on Linux/Mac)
python download_models.py --text
```

Set `TEXT_DETECTOR_MODEL` or `IMAGE_DETECTOR_MODEL` to `none` to switch a pre-trained detector off.
Most standard Hugging Face text/image classifiers work; check the backend log after switching.

**Which model the backend uses:** a downloaded pre-trained detector → your own trained models
(`models/saved/`) → the built-in fallback.

**Good to know**
- The accuracy figures above are the model authors' own results, not measured in this project.
  Measure on your own held-out data before quoting numbers in a report.
- All detectors get weaker on very short text, heavily edited or paraphrased text, and on AI models
  released after the detector was trained.
- SHAP explanations work with every text model. GradCAM heatmaps come only from the project's own
  EfficientNet; the pre-trained image detector gives a verdict without a heatmap.
- Docker: `docker compose exec backend python download_models.py` (kept in the `hf-cache` volume).
- To free the space again, delete the `models--…` folders in the Hugging Face cache
  (`~/.cache/huggingface/hub`, on Windows `C:\Users\<you>\.cache\huggingface\hub`).

## 🧠 Model Training

### Step 1 — Prepare Datasets

```bash
cd training
pip install datasets pillow

python dataset_prep.py --output_dir ./data
```

This will download and prepare:
- HC3 Dataset (AI vs Human text)
- CIFAKE (AI vs real images)

### Step 2 — Train Text Models

```bash
# Train RoBERTa (AI text detector)
python train_roberta.py \
    --data_path ./data/text/ai_human.jsonl \
    --output_dir ../backend/models/saved/roberta_ai_detector \
    --epochs 5 --batch_size 16 --lr 2e-5

# Train LinguaForge-X (BERT + linguistic features)
python train_linguaforge.py \
    --data_path ./data/text/ai_human.jsonl \
    --output_dir ../backend/models/saved/linguaforge_detector \
    --epochs 5 --batch_size 16

# Train RoBERTa + LinguaForge-X Ensemble
python train_ensemble.py \
    --roberta_dir ../backend/models/saved/roberta_ai_detector \
    --lingua_dir ../backend/models/saved/linguaforge_detector \
    --data_path ./data/text/ai_human.jsonl
```

### Step 3 — Train Image Models

```bash
# EfficientNet-B4
python train_image_classifier.py \
    --data_dir ./data/images \
    --model efficientnet \
    --epochs 10 --batch_size 32

# Vision Transformer (ViT-B/16)
python train_image_classifier.py \
    --data_dir ./data/images \
    --model vit \
    --epochs 10 --batch_size 32
```

### Expected Model Performance

| Model | Task | Accuracy | F1 |
|---|---|---|---|
| RoBERTa | AI Text Detection | 95.8% | 0.956 |
| LinguaForge-X | AI Text Detection | 94.6% | 0.944 |
| **Ensemble (R+L)** | **AI Text Detection** | **97.4%** | **0.972** |
| EfficientNet-B4 | AI Image | 96.2% | 0.960 |
| ViT-B/16 | AI Image | 95.8% | 0.956 |

---

## 🔌 API Reference

### Text Analysis
```
POST /api/text/analyze
Body: { "text": "Your text here...", "explain": true }   // explain=false skips SHAP (faster)

Response:
{
  "prediction": "AI Generated",
  "ai_confidence": 0.8742,
  "human_confidence": 0.1258,
  "model_used": "RoBERTa + LinguaForge-X ensemble",
  "roberta": { "prediction": "AI Generated", "ai_confidence": 0.8912 },      // null if not loaded
  "linguaforge": { "prediction": "AI Generated", "ai_confidence": 0.8571 },  // null if not loaded
  "ensemble_strategy": "weighted_average",
  "ensemble_weight": { "roberta": 0.55, "linguaforge": 0.45 },
  "token_importance": [{ "token": "...", "shap": 0.105, "score": 0.76, "label": "ai" }],
  "shap_values": [{ "feature": "...", "shap": 0.105 }],
  "suspicious_phrases": [...],
  "processing_time_ms": 412
}
```

### Image Analysis
```
POST /api/image/analyze
Body: multipart/form-data (file: JPG, PNG, WEBP or GIF image)

Response:
{
  "prediction": "AI Generated",
  "ai_confidence": 0.9123,
  "real_confidence": 0.0877,
  "model_used": "EfficientNet-B4 + ViT-B/16 ensemble",
  "efficientnet": { "prediction": "AI Generated", "confidence": 0.9201 },  // null in heuristic mode
  "vit": { "prediction": "AI Generated", "confidence": 0.9045 },           // null in heuristic mode
  "gradcam_regions": [...],
  "signals": [...],  // heuristic checks that fired (heuristic mode only)
  "processing_time_ms": 847
}
```

### Dashboard & History
```
GET    /api/stats?days=7              # totals + per-day series for the last 7 (or 30) days
GET    /api/history?limit=10&offset=0 # recorded analyses, newest first
DELETE /api/history                   # delete all recorded analyses
```

---

## 🐳 Docker Deployment

```bash
# Build and run entire stack
docker-compose up --build

# Frontend: http://localhost:5173
# Backend:  http://localhost:8000
```

---

## 📊 Datasets

| Dataset | Type | Samples | Source |
|---|---|---|---|
| HC3 | AI vs Human Text | 58,000+ | HuggingFace |
| GPT-Wiki-Intro | AI Generated | 150,000+ | HuggingFace |
| OpenAI Human vs AI | Classification | 500,000+ | OpenAI |
| CIFAKE | AI vs Real Images | 120,000 | HuggingFace |
| DiffusionDB | Stable Diffusion | 2,000,000+ | HuggingFace |

---

## 🛠️ Tech Stack

**Frontend:** React 18 · Vite · Tailwind CSS · Framer Motion · Recharts · Axios

**Backend:** FastAPI · Python 3.11 · PyTorch 2.2 · Transformers 4.40

**Models:** RoBERTa · LinguaForge-X · BERT · EfficientNet-B4 · ViT-B/16

**XAI:** SHAP · GradCAM

**Database:** SQLite (analysis history for the dashboard, `backend/data/history.db`)

---

## 🌏 SDG Alignment

This project aligns with **UN Sustainable Development Goal 16** —
_Peace, Justice and Strong Institutions_ — by providing tools to detect and
counter AI-generated misinformation that undermines democratic processes,
public trust, and social cohesion.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

*Built with  as a Final Year Major Project — CSE (Data Science)*
