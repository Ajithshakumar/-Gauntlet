"""
PreFlight AI - Model Predictor
==============================
Loads pretrained CV models and executes inference with proper preprocessing.
Supports ResNet50, MobileNetV2, EfficientNet-B0 from torchvision.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import io

from config import IMAGENET_CLASSES


class ModelPredictor:
    """Handles model loading, preprocessing, and inference."""

    # ImageNet normalization constants
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])

    # Standard preprocessing pipeline
    PREPROCESS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    AVAILABLE_MODELS = {
        "resnet50": {
            "name": "ResNet-50",
            "description": "Deep residual network, 25.6M parameters",
            "loader": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2),
            "accuracy": "80.86%",
        },
        "resnet18": {
            "name": "ResNet-18",
            "description": "Lightweight residual network, 11.7M parameters",
            "loader": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
            "accuracy": "69.76%",
        },
        "mobilenet_v2": {
            "name": "MobileNet-V2",
            "description": "Efficient mobile architecture, 3.5M parameters",
            "loader": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2),
            "accuracy": "72.15%",
        },
        "efficientnet_b0": {
            "name": "EfficientNet-B0",
            "description": "Compound scaled network, 5.3M parameters",
            "loader": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            "accuracy": "77.69%",
        },
    }

    def __init__(self, model_name: str = "resnet50", device: str = None):
        """Initialize predictor with specified model."""
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_name: str) -> nn.Module:
        """Load pretrained model."""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}")

        model_info = self.AVAILABLE_MODELS[model_name]
        print(f"Loading {model_info['name']}...")
        model = model_info["loader"]()
        return model

    @classmethod
    def preprocess_image(cls, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to preprocessed tensor."""
        # Convert BGR to RGB if needed
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        pil_image = Image.fromarray(image_rgb)
        tensor = cls.PREPROCESS(pil_image)
        return tensor

    @classmethod
    def denormalize_image(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse ImageNet normalization for visualization."""
        mean = cls.MEAN.view(3, 1, 1).to(tensor.device)
        std = cls.STD.view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    def predict(self, image: np.ndarray) -> Dict:
        """Run inference on a single image."""
        tensor = self.preprocess_image(image).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get top-5 predictions
        probs, indices = torch.topk(probabilities[0], 5)

        results = {
            "predicted_class": IMAGENET_CLASSES[indices[0].item()],
            "confidence": probs[0].item(),
            "top5": [
                {"class": IMAGENET_CLASSES[idx.item()], "confidence": prob.item()}
                for idx, prob in zip(indices, probs)
            ],
            "class_index": indices[0].item(),
        }

        return results

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Run inference on a batch of images."""
        tensors = torch.stack([self.preprocess_image(img) for img in images])
        tensors = tensors.to(self.device)

        with torch.no_grad():
            outputs = self.model(tensors)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        results = []
        for i in range(len(images)):
            probs, indices = torch.topk(probabilities[i], 5)
            results.append({
                "predicted_class": IMAGENET_CLASSES[indices[0].item()],
                "confidence": probs[0].item(),
                "top5": [
                    {"class": IMAGENET_CLASSES[idx.item()], "confidence": prob.item()}
                    for idx, prob in zip(indices, probs)
                ],
                "class_index": indices[0].item(),
            })

        return results

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        info = self.AVAILABLE_MODELS.get(self.model_name, {})
        return {
            "name": info.get("name", self.model_name),
            "description": info.get("description", ""),
            "accuracy": info.get("accuracy", "N/A"),
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }
