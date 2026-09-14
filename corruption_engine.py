"""
PreFlight AI - Corruption Engine (FIXED)
=========================================
Adds Rain Streaks corruption to match patent claims.
All other corruptions unchanged for compatibility.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import random

from config import CORRUPTIONS, SEVERITY_LEVELS, AgentProfile


class CorruptionEngine:
    """Engine for applying structured image corruptions."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _severity_to_param(self, corruption_name: str, severity: int) -> float:
        """Map severity level 1-5 to actual parameter value."""
        config = CORRUPTIONS[corruption_name]
        min_val, max_val = config.severity_range
        return min_val + (max_val - min_val) * ((severity - 1) / 4.0)

    def apply_corruption(self, image: np.ndarray, corruption_name: str,
                        severity: int, model=None, device='cpu') -> np.ndarray:
        """Apply a specific corruption to an image."""
        param = self._severity_to_param(corruption_name, severity)

        if corruption_name == "gaussian_noise":
            return self._gaussian_noise(image, param)
        elif corruption_name == "motion_blur":
            return self._motion_blur(image, int(param))
        elif corruption_name == "fog":
            return self._fog(image, param)
        elif corruption_name == "rain":
            # Map severity 1-5 to light/med/heavy
            density_map = {1: "light", 2: "light", 3: "med", 4: "med", 5: "heavy"}
            density = density_map.get(severity, "med")
            return self._rain(image, density)
        elif corruption_name == "brightness":
            return self._brightness(image, param)
        elif corruption_name == "jpeg_compression":
            return self._jpeg_compression(image, int(param))
        elif corruption_name == "pixelation":
            return self._pixelation(image, int(param))
        elif corruption_name == "gaussian_blur":
            return self._gaussian_blur(image, param)
        elif corruption_name == "occlusion":
            return self._occlusion(image, param)
        elif corruption_name == "rotation":
            return self._rotation(image, param)
        elif corruption_name == "fgsm":
            if model is None:
                return image
            return self._fgsm_attack(image, model, device, param)
        elif corruption_name == "pgd":
            if model is None:
                return image
            return self._pgd_attack(image, model, device, param)
        else:
            return image

    def _gaussian_noise(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, sigma * 255, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _motion_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply motion blur with random direction."""
        kernel_size = max(3, kernel_size | 1)
        kernel = np.zeros((kernel_size, kernel_size))
        angle = random.uniform(0, 180)
        center = kernel_size // 2
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * np.cos(np.radians(angle)))
            y = int(center + offset * np.sin(np.radians(angle)))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        kernel = kernel / kernel.sum()
        return cv2.filter2D(image, -1, kernel)

    def _fog(self, image: np.ndarray, alpha: float) -> np.ndarray:
        """Simulate fog/haze with alpha blending."""
        fog_layer = np.ones_like(image) * 200
        foggy = cv2.addWeighted(image.astype(np.float32), 1 - alpha,
                                fog_layer.astype(np.float32), alpha, 0)
        return foggy.astype(np.uint8)

    def _rain(self, image: np.ndarray, density: str) -> np.ndarray:
        """Simulate rain streaks on image. Patent-aligned weather corruption."""
        result = image.copy()
        h, w = image.shape[:2]

        # Number of raindrops based on density
        density_map = {"light": 800, "med": 2500, "heavy": 5000}
        num_drops = density_map.get(density, 2500)

        for _ in range(num_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(8, 25)
            thickness = np.random.randint(1, 2)
            # Slight angle for realism
            angle_offset = np.random.randint(-3, 3)
            # Rain color: light gray/white with transparency effect
            color = (200, 200, 210)
            cv2.line(result, (x, y), (x + angle_offset, y + length), color, thickness)

        # Blend with original for transparency effect
        result = cv2.addWeighted(image, 0.85, result, 0.15, 0)
        return result

    def _brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness."""
        if random.random() > 0.5:
            factor = -factor
        adjusted = image.astype(np.float32) * (1 + factor)
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _jpeg_compression(self, image: np.ndarray, quality: int) -> np.ndarray:
        """Apply JPEG compression artifacts."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, 1)
        return decoded

    def _pixelation(self, image: np.ndarray, block_size: int) -> np.ndarray:
        """Apply pixelation by downsampling and upsampling."""
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // block_size, h // block_size),
                          interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian blur."""
        ksize = int(sigma * 3) * 2 + 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def _occlusion(self, image: np.ndarray, ratio: float) -> np.ndarray:
        """Apply random rectangular occlusion."""
        h, w = image.shape[:2]
        occ_h = int(h * np.sqrt(ratio))
        occ_w = int(w * np.sqrt(ratio))
        y = random.randint(0, max(0, h - occ_h))
        x = random.randint(0, max(0, w - occ_w))
        occluded = image.copy()
        occluded[y:y+occ_h, x:x+occ_w] = np.random.randint(0, 255, (occ_h, occ_w, 3))
        return occluded

    def _rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image and crop to original size."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(128, 128, 128))
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return rotated[start_y:start_y+h, start_x:start_x+w]

    def _fgsm_attack(self, image: np.ndarray, model, device: str,
                     epsilon: float) -> np.ndarray:
        """Fast Gradient Sign Method adversarial attack."""
        from model_predictor import ModelPredictor
        img_tensor = ModelPredictor.preprocess_image(image).to(device).clone()
        img_tensor.requires_grad = True
        model.eval()
        output = model(img_tensor.unsqueeze(0))
        pred_class = output.argmax(dim=1).item()
        loss = F.cross_entropy(output, torch.tensor([pred_class]).to(device))
        model.zero_grad()
        loss.backward()
        grad_sign = img_tensor.grad.data.sign()
        perturbed = img_tensor + epsilon * grad_sign
        perturbed = ModelPredictor.denormalize_image(perturbed)
        perturbed = torch.clamp(perturbed, 0, 1)
        return (perturbed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def _pgd_attack(self, image: np.ndarray, model, device: str,
                    epsilon: float) -> np.ndarray:
        """Projected Gradient Descent adversarial attack."""
        from model_predictor import ModelPredictor
        img_tensor = ModelPredictor.preprocess_image(image).to(device)
        original = img_tensor.clone()
        model.eval()
        pred_class = model(img_tensor.unsqueeze(0)).argmax(dim=1).item()
        step_size = epsilon / 4
        num_steps = 10
        perturbed = img_tensor.clone()
        perturbed.requires_grad = True
        for _ in range(num_steps):
            output = model(perturbed.unsqueeze(0))
            loss = F.cross_entropy(output, torch.tensor([pred_class]).to(device))
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                perturbed = perturbed + step_size * perturbed.grad.sign()
                perturbation = torch.clamp(perturbed - original, -epsilon, epsilon)
                perturbed = torch.clamp(original + perturbation, 0, 1)
            perturbed.requires_grad = True
        perturbed = ModelPredictor.denormalize_image(perturbed)
        perturbed = torch.clamp(perturbed, 0, 1)
        return (perturbed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    def generate_corruption_grid(self, image: np.ndarray,
                                  corruption_names: List[str],
                                  severities: List[int] = None,
                                  model=None, device='cpu') -> Dict:
        """Generate all corrupted variants for given corruptions and severities."""
        if severities is None:
            severities = SEVERITY_LEVELS
        grid = {"original": image, "corruptions": {}}
        for name in corruption_names:
            grid["corruptions"][name] = {}
            for sev in severities:
                corrupted = self.apply_corruption(image, name, sev, model, device)
                grid["corruptions"][name][sev] = corrupted
        return grid

    def get_agent_corruptions(self, agent: AgentProfile,
                              top_n: int = 6) -> List[str]:
        """Get corruption types sorted by agent weights."""
        weights = agent.corruption_weights
        sorted_corruptions = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_corruptions[:top_n]]