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

- **NLP (RoBERTa + LinguaForge-X)** — AI-generated text & fake news detection
- **Computer Vision (EfficientNet + ViT)** — AI-generated image detection
- **Explainable AI (SHAP, LIME, GradCAM)** — Full reasoning transparency
- **Ensemble Learning** — Weighted prediction fusion for maximum accuracy

Every prediction includes **full explainability on the same dashboard** — no separate pages.

---

## 🏗️ Project Structure

```
DeepFakeShield/
├── frontend/                    # React + Vite + Tailwind + Framer Motion
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.jsx      # Hero, features, stats, pipeline, CTA
│   │   │   ├── Dashboard.jsx    # Analytics, charts, recent analyses
│   │   │   ├── TextAnalysis.jsx # Input → verdict + XAI + SHAP + ensemble
│   │   │   ├── ImageAnalysis.jsx# Upload → verdict + GradCAM + model compare
│   │   │   └── About.jsx        # Models, datasets, tech stack, SDG
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   └── Footer.jsx
│   │   └── utils/api.js         # Axios API calls
│   └── package.json
│
├── backend/                     # FastAPI + PyTorch
│   ├── main.py                  # App entry + CORS
│   ├── routers/
│   │   ├── text.py              # /api/text/analyze, /api/text/fake-news
│   │   ├── image.py             # /api/image/analyze
│   │   └── stats.py             # /api/stats
│   ├── services/
│   │   ├── text_service.py      # RoBERTa + LinguaForge-X ensemble
│   │   └── image_service.py     # EfficientNet + ViT + GradCAM
│   ├── models/saved/            # Trained model weights (after training)
│   └── requirements.txt
│
├── training/                    # Model training scripts
│   ├── dataset_prep.py          # Download + prepare all datasets
│   ├── train_roberta.py         # Fine-tune RoBERTa for AI text detection
│   ├── train_linguaforge.py     # Train LinguaForge-X (BERT + linguistic features)
│   ├── train_ensemble.py        # Grid search + meta-learner ensemble fusion
│   ├── train_fakenews.py        # BERT fake news classifier
│   └── train_image_classifier.py# EfficientNet / ViT image classifier
│
├── docs/                        # Documentation
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
git clone https://github.com/yourusername/DeepFakeShield.git
cd DeepFakeShield
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

> **Note:** The backend works **without trained models** using realistic mock inference.
> Train your models (see Training section) to enable real AI inference.

---

## 🧠 Model Training

### Step 1 — Prepare Datasets

```bash
cd training
pip install datasets pillow

python dataset_prep.py --output_dir ./data
```

This will download and prepare:
- HC3 Dataset (AI vs Human text)
- LIAR Dataset (fake news)
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

# Train Fake News Classifier
python train_fakenews.py \
    --data_path ./data/text/fakenews.jsonl \
    --output_dir ../backend/models/saved/fakenews_classifier \
    --epochs 5
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
| BERT | Fake News | 93.2% | 0.930 |
| EfficientNet-B4 | AI Image | 96.2% | 0.960 |
| ViT-B/16 | AI Image | 95.8% | 0.956 |

---

## 🔌 API Reference

### Text Analysis
```
POST /api/text/analyze
Body: { "text": "Your text here..." }

Response:
{
  "prediction": "AI Generated",
  "ai_confidence": 0.8742,
  "human_confidence": 0.1258,
  "fake_news_score": 0.5431,
  "roberta": { "prediction": "AI Generated", "ai_confidence": 0.8912 },
  "linguaforge": { "prediction": "AI Generated", "ai_confidence": 0.8571 },
  "ensemble_weight": { "roberta": 0.55, "linguaforge": 0.45 },
  "token_importance": [...],
  "shap_values": [...],
  "suspicious_phrases": [...],
  "processing_time_ms": 412
}
```

### Image Analysis
```
POST /api/image/analyze
Body: multipart/form-data (file: image)

Response:
{
  "prediction": "AI Generated",
  "ai_confidence": 0.9123,
  "real_confidence": 0.0877,
  "efficientnet": { "prediction": "AI Generated", "confidence": 0.9201 },
  "vit": { "prediction": "AI Generated", "confidence": 0.9045 },
  "gradcam_regions": [...],
  "artifacts_detected": [...],
  "processing_time_ms": 847
}
```

### Dashboard Stats
```
GET /api/stats
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
| LIAR | Fake News (6-class) | 12,791 | HuggingFace |
| GPT-Wiki-Intro | AI Generated | 150,000+ | HuggingFace |
| OpenAI Human vs AI | Classification | 500,000+ | OpenAI |
| CIFAKE | AI vs Real Images | 120,000 | HuggingFace |
| DiffusionDB | Stable Diffusion | 2,000,000+ | HuggingFace |

---

## 🛠️ Tech Stack

**Frontend:** React 18 · Vite · Tailwind CSS · Framer Motion · Recharts · Axios

**Backend:** FastAPI · Python 3.11 · PyTorch 2.2 · Transformers 4.40

**Models:** RoBERTa · LinguaForge-X · BERT · EfficientNet-B4 · ViT-B/16

**XAI:** SHAP · LIME · GradCAM

**Database:** MongoDB (optional, for analysis history)

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

*Built with ❤️ as a Final Year Major Project — CSE (Data Science)*
