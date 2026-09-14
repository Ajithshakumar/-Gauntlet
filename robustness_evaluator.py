"""
PreFlight AI - Robustness Evaluator (FIXED)
============================================
Deterministic systematic testing across ALL corruptions.
Fixed scoring math: no negative values, clamped to [0, 100].
Agent parameter accepted but ignored — all corruptions run systematically.
"""

import numpy as np
from typing import Dict, List, Optional

from corruption_engine import CorruptionEngine
from model_predictor import ModelPredictor
from config import CORRUPTIONS, SEVERITY_LEVELS, CORE_CORRUPTION_KEYS


class RobustnessEvaluator:
    """Main evaluation orchestrator — DETERMINISTIC, no agent gatekeeping."""

    def __init__(self, model_name: str = "resnet50", device: str = None):
        self.predictor = ModelPredictor(model_name, device)
        self.engine = CorruptionEngine()
        self.results = []

    def evaluate_image(self, image: np.ndarray,
                       agent=None,  # ACCEPTED FOR BACKWARD COMPAT, IGNORED
                       severities: List[int] = None,
                       progress_callback=None) -> Dict:
        """Evaluate robustness — runs ALL core corruptions systematically."""
        if severities is None:
            severities = [1, 2, 3]  # Patent-aligned default

        # Baseline prediction
        baseline = self.predictor.predict(image)
        baseline_class = baseline["predicted_class"]
        baseline_conf = baseline["confidence"]

        results = {
            "baseline": baseline,
            "corruptions": {},
            "metrics": {},
        }

        # DETERMINISTIC: always run the 6 core corruptions from patent
        # If CORE_CORRUPTION_KEYS not in config, fall back to all corruptions
        try:
            corruption_names = CORE_CORRUPTION_KEYS
        except Exception:
            corruption_names = list(CORRUPTIONS.keys())

        total_tests = len(corruption_names) * len(severities)
        test_count = 0

        for corr_name in corruption_names:
            results["corruptions"][corr_name] = {}

            for sev in severities:
                # Apply corruption
                corrupted = self.engine.apply_corruption(
                    image, corr_name, sev,
                    model=self.predictor.model,
                    device=self.predictor.device
                )

                # Predict on corrupted image
                pred = self.predictor.predict(corrupted)

                # Store result
                results["corruptions"][corr_name][sev] = {
                    "predicted_class": pred["predicted_class"],
                    "confidence": pred["confidence"],
                    "top5": pred["top5"],
                    "class_changed": pred["predicted_class"] != baseline_class,
                    "confidence_drop": baseline_conf - pred["confidence"],
                    "top5_retained": baseline_class in [t["class"] for t in pred["top5"]],
                }

                test_count += 1
                if progress_callback:
                    progress_callback(test_count / total_tests)

        # Compute metrics
        results["metrics"] = self._compute_metrics(results, baseline)

        return results

    def _compute_metrics(self, results: Dict, baseline: Dict) -> Dict:
        """Compute robustness metrics — FIXED MATH, NO NEGATIVES."""
        corruptions = results["corruptions"]
        baseline_class = baseline["predicted_class"]
        baseline_conf = baseline["confidence"]

        total_tests = 0
        stable_count = 0
        confidences = []
        top5_retained = 0

        for corr_name, severities in corruptions.items():
            for sev, result in severities.items():
                total_tests += 1
                if not result["class_changed"]:
                    stable_count += 1
                confidences.append(result["confidence"])
                if result["top5_retained"]:
                    top5_retained += 1

        # Stability: % of tests where class didn't change
        stability = (stable_count / total_tests * 100) if total_tests > 0 else 0.0

        # Confidence retention: avg corrupted confidence vs baseline
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        if baseline_conf > 0:
            confidence_retention = (avg_confidence / baseline_conf) * 100.0
        else:
            confidence_retention = 0.0

        # CRITICAL FIX: cap retention at 100 so HM never exceeds 100
        confidence_retention = min(100.0, confidence_retention)
        stability = min(100.0, stability)

        # Harmonic mean of stability and retention
        if stability + confidence_retention > 0:
            harmonic_mean = 2.0 * (stability * confidence_retention) / (stability + confidence_retention)
        else:
            harmonic_mean = 0.0

        # CRITICAL FIX: cap harmonic mean
        harmonic_mean = min(100.0, harmonic_mean)

        # Risk score: inverse of harmonic mean, adjusted by top-5 degradation
        top5_accuracy = (top5_retained / total_tests * 100) if total_tests > 0 else 0.0
        risk_score = (1.0 - harmonic_mean / 100.0) * 100.0
        risk_score = risk_score * (1.0 + (1.0 - top5_accuracy / 100.0) * 0.2)

        # CRITICAL FIX: clamp to [0, 100]
        risk_score = float(np.clip(risk_score, 0.0, 100.0))

        return {
            "stability": round(stability, 2),
            "confidence_retention": round(confidence_retention, 2),
            "top5_accuracy": round(top5_accuracy, 2),
            "harmonic_mean": round(harmonic_mean, 2),
            "risk_score": round(risk_score, 2),
            "total_tests": total_tests,
        }

    def evaluate_batch(self, images: List[np.ndarray],
                       agent=None,
                       severities: List[int] = None,
                       progress_callback=None) -> List[Dict]:
        """Evaluate a batch of images."""
        all_results = []
        for i, image in enumerate(images):
            result = self.evaluate_image(image, agent, severities, progress_callback)
            all_results.append(result)
        return all_results