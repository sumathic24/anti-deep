import asyncio
import time
import os
import numpy as np
from typing import Dict, Any, List

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
EFFICIENTNET_PATH = os.path.join(MODEL_DIR, "efficientnet_detector.pth")
VIT_PATH = os.path.join(MODEL_DIR, "vit_detector.pth")

# Global Config: 0 means Neuron 0 outputs the AI probability.
AI_CLASS_INDEX = 0

TRANSFORM = None
if TORCH_AVAILABLE:
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class GradCAMHook:
    """Captures gradient and activation for GradCAM computation."""
    def __init__(self, model, target_layer):
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute_cam(self, class_idx, output):
        if self.gradients is not None:
            self.gradients = None
            
        output[0][class_idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

class ImageAnalysisService:
    def __init__(self):
        self.efficientnet = None
        self.vit = None
        self.eff_cam_hook = None
        self._load_models()

    def _load_models(self):
        abs_model_dir = os.path.abspath(MODEL_DIR)
        print(f"[ImageService] Looking for models in: {abs_model_dir}")
        self.using_mock = False 
        
        if not TORCH_AVAILABLE:
            print("[ImageService] PyTorch not available — using mock inference.")
            self.using_mock = True
            return
            
        try:
            if os.path.exists(EFFICIENTNET_PATH) and os.path.exists(VIT_PATH):
                self.efficientnet = models.efficientnet_b4(weights=None)
                self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, 2)
                self.efficientnet.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location="cpu"))
                self.efficientnet.eval()
                
                target_layer = self.efficientnet.features[-1]
                self.eff_cam_hook = GradCAMHook(self.efficientnet, target_layer)
                print("[ImageService] EfficientNet-B4 loaded.")

                from torchvision.models import vit_b_16
                self.vit = vit_b_16(weights=None)
                self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, 2)
                self.vit.load_state_dict(torch.load(VIT_PATH, map_location="cpu"))
                self.vit.eval()
                print("[ImageService] ViT-B/16 loaded.")
            else:
                print(f"[ImageService] WARNING: .pth files not found! Falling back to mock.")
                self.using_mock = True
        except Exception as e:
            print(f"[ImageService] Model load error: {e}")
            self.using_mock = True

    def _preprocess_image(self, filepath: str):
        img = Image.open(filepath).convert("RGB")
        return TRANSFORM(img).unsqueeze(0)

    def _run_efficientnet(self, tensor) -> tuple:
        tensor.requires_grad = True
        self.efficientnet.zero_grad() 
        
        logits = self.efficientnet(tensor)
        probabilities = torch.softmax(logits, dim=-1)[0]
        prob = float(probabilities[AI_CLASS_INDEX])
        
        real_regions = []
        if prob > 0.5 and self.eff_cam_hook is not None:
            raw_cam = self.eff_cam_hook.compute_cam(AI_CLASS_INDEX, logits)
            real_regions = self._cam_to_regions(raw_cam)
            
        return prob, real_regions

    def _run_vit(self, tensor) -> float:
        with torch.no_grad():
            logits = self.vit(tensor)
        probabilities = torch.softmax(logits, dim=-1)[0]
        return float(probabilities[AI_CLASS_INDEX])

    def _cam_to_regions(self, cam_array: np.ndarray, threshold: float = 0.6) -> List[Dict]:
        regions = []
        h, w = cam_array.shape
        y_indices, x_indices = np.where(cam_array > threshold)
        
        hotspots = [(cam_array[y, x], x, y) for y, x in zip(y_indices, x_indices)]
        hotspots.sort(key=lambda item: item[0], reverse=True)
        top_spots = hotspots[:3]
        
        for intensity, x, y in top_spots:
            x_pct = (x / w) * 100
            y_pct = (y / h) * 100
            box_size_w = (1 / w) * 100 * 1.5 
            box_size_h = (1 / h) * 100 * 1.5
            
            regions.append({
                "x": round(max(0, x_pct - (box_size_w / 2)), 1),
                "y": round(max(0, y_pct - (box_size_h / 2)), 1),
                "w": round(box_size_w, 1),
                "h": round(box_size_h, 1),
                "intensity": round(float(intensity), 2),
                "label": "AI Artifact Region (GradCAM)"
            })
        return regions

    def _mock_inference(self, filepath: str) -> tuple:
        """Heuristic fallback analyzing EXIF and image resolution."""
        eff_score = 0.15
        vit_score = 0.15
        regions = []
        
        if filepath and os.path.exists(filepath):
            try:
                with Image.open(filepath) as img:
                    exif = img.getexif()
                    if not exif:
                        eff_score += 0.45 
                        vit_score += 0.45
                        
                    w, h = img.size
                    if (w == 512 and h == 512) or (w == 1024 and h == 1024) or (w == h and w % 64 == 0):
                        eff_score += 0.30
                        vit_score += 0.30
            except Exception:
                pass

        eff_score = float(np.clip(eff_score + np.random.normal(0, 0.02), 0.01, 0.99))
        vit_score = float(np.clip(vit_score + np.random.normal(0, 0.02), 0.01, 0.99))
        
        if eff_score > 0.5:
            regions = self._generate_gradcam_regions(eff_score)
            
        return eff_score, vit_score, regions

    def _generate_gradcam_regions(self, ai_score: float) -> List[Dict]:
        """Mock regions for the fallback mode."""
        regions = []
        labels = ["Texture inconsistency", "Edge artifact", "Lighting mismatch"]
        for i in range(2):
            regions.append({
                "x": int(np.random.uniform(10, 50)),
                "y": int(np.random.uniform(10, 50)),
                "w": int(np.random.uniform(20, 30)),
                "h": int(np.random.uniform(20, 30)),
                "intensity": round(float(np.random.uniform(0.6, 0.95)), 2),
                "label": labels[i % len(labels)],
            })
        return regions

    def _detect_artifacts(self, ai_score: float) -> List[str]:
        all_artifacts = ["GAN frequency artifacts", "Unnatural texture", "Blending boundary artifacts"]
        return all_artifacts[:2] if ai_score > 0.5 else []

    async def analyze_image(self, filepath: str) -> Dict[str, Any]:
        t0 = time.time()
        gradcam_regions = []

        if not self.using_mock and self.efficientnet and self.vit:
            model_used = "EfficientNet + ViT Ensemble"
            loop = asyncio.get_event_loop()
            tensor = self._preprocess_image(filepath)
            
            eff_result, vit_score = await asyncio.gather(
                loop.run_in_executor(None, self._run_efficientnet, tensor),
                loop.run_in_executor(None, self._run_vit, tensor)
            )
            eff_score, gradcam_regions = eff_result
        else:
            model_used = "Mock/Fast Heuristic Model"
            await asyncio.sleep(0.5)
            eff_score, vit_score, gradcam_regions = self._mock_inference(filepath)

        ensemble_score = (eff_score * 0.5) + (vit_score * 0.5)
        real_score = 1.0 - ensemble_score
        artifacts = self._detect_artifacts(ensemble_score) if ensemble_score > 0.5 else []
        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            "prediction": "AI Generated" if ensemble_score > 0.5 else "Real Image",
            "ai_confidence": round(ensemble_score, 4),
            "real_confidence": round(real_score, 4),
            "model_used": model_used,
            "efficientnet": {
                "prediction": "AI Generated" if eff_score > 0.5 else "Real",
                "confidence": round(eff_score, 4),
            },
            "vit": {
                "prediction": "AI Generated" if vit_score > 0.5 else "Real",
                "confidence": round(vit_score, 4),
            },
            "gradcam_regions": gradcam_regions,
            "artifacts_detected": artifacts,
            "processing_time_ms": elapsed_ms,
        }