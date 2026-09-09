from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from routers import text, image, stats

app = FastAPI(
    title="DeepFakeShield API",
    description="Multimodal AI detection platform for deepfakes and misinformation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount upload directory
os.makedirs("uploads", exist_ok=True)

app.include_router(text.router, prefix="/api/text", tags=["Text Analysis"])
app.include_router(image.router, prefix="/api/image", tags=["Image Analysis"])
app.include_router(stats.router, prefix="/api", tags=["Stats"])


@app.get("/")
async def root():
    return {
        "name": "DeepFakeShield API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
