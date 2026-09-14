"""
PreFlight AI - Agent System (FIXED)
====================================
Scenario selection is USER-DRIVEN, not fake AI detection.
ImageAnalyzer kept for backward compatibility but detection is disabled.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import AGENTS, AgentProfile, CORRUPTIONS


class ImageAnalyzer:
    """Analyzes image characteristics — kept for backward compat. Detection DISABLED."""

    def __init__(self):
        pass

    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """Extract image features and return scenario scores."""
        features = {}

        # Brightness analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)

        features["mean_brightness"] = mean_brightness
        features["brightness_std"] = brightness_std
        features["is_dark"] = mean_brightness < 80
        features["is_bright"] = mean_brightness > 180
        features["is_low_contrast"] = brightness_std < 30

        # Edge analysis (motion/blur detection)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features["edge_density"] = edge_density
        features["is_blurry"] = edge_density < 0.02

        # Texture analysis
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features["texture_variance"] = laplacian_var
        features["is_smooth"] = laplacian_var < 100

        # Color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        features["mean_saturation"] = mean_saturation
        features["is_colorful"] = mean_saturation > 80

        # Detect potential objects (simple contour analysis)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features["num_contours"] = len(contours)
        features["has_multiple_objects"] = len(contours) > 5

        # Aspect ratio
        h, w = image.shape[:2]
        features["aspect_ratio"] = w / h
        features["is_landscape"] = w > h
        features["is_portrait"] = h > w

        return features

    def detect_scenario(self, image: np.ndarray) -> Dict[str, float]:
        """DEPRECATED: Returns uniform low scores. Detection is user-driven now."""
        # Old heuristic detection was unreliable (e.g., warplane detected as medical).
        # Now returns uniform scores so UI still renders, but agent is user-selected.
        return {key: 0.1 for key in AGENTS.keys()}


class AgentSelector:
    """Selects the best agent based on USER CHOICE, not fake image analysis."""

    def __init__(self):
        self.analyzer = ImageAnalyzer()

    def select_agent(self, image: np.ndarray, user_preference: str = None) -> Tuple[AgentProfile, Dict]:
        """Select the best agent for the given image — USER DRIVEN."""

        # If user picked something valid, use it
        if user_preference and user_preference in AGENTS:
            agent = AGENTS[user_preference]
            reasoning = {
                "selected_agent": agent.name,
                "confidence": 1.0,
                "scenario_scores": {k: 0.1 for k in AGENTS.keys()},
                "features": self.analyzer.analyze(image),
                "reasoning": f"User selected: {agent.name}. All 18 corruption variants will be tested.",
            }
            return agent, reasoning

        # Default: general purpose, no fake detection
        agent = AGENTS["general"]
        reasoning = {
            "selected_agent": agent.name,
            "confidence": 1.0,
            "scenario_scores": {k: 0.1 for k in AGENTS.keys()},
            "features": self.analyzer.analyze(image),
            "reasoning": "General-purpose robustness testing. All 18 corruption variants will be applied systematically regardless of image content.",
        }
        return agent, reasoning

    def get_all_agents(self) -> List[Dict]:
        """Get all available agents with descriptions."""
        return [
            {
                "id": key,
                "name": agent.name,
                "description": agent.description,
                "icon": agent.icon,
            }
            for key, agent in AGENTS.items()
        ]