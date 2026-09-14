"""
PreFlight AI - Metrics Analyzer (FIXED)
==========================================
Smarter recommendations: weighs overall Φ score, not just per-corruption panic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from io import BytesIO

from config import RISK_THRESHOLDS, CORRUPTIONS


class MetricsAnalyzer:
    """Analyzes evaluation results and generates insights."""

    def __init__(self):
        pass

    def compute_risk_category(self, risk_score: float) -> str:
        """Categorize risk score."""
        if risk_score <= 30:
            return "low"
        elif risk_score <= 60:
            return "medium"
        return "high"

    def generate_heatmap_data(self, results: Dict) -> pd.DataFrame:
        """Generate heatmap data from corruption results."""
        corruptions = results["corruptions"]
        data = {}
        for corr_name, severities in corruptions.items():
            row = {}
            for sev, result in severities.items():
                row[sev] = result["confidence"] * 100
            data[corr_name] = row

        df = pd.DataFrame(data).T
        df.columns = [f"Severity {c}" for c in df.columns]
        df.index = [
            CORRUPTIONS[idx].name if idx in CORRUPTIONS else idx
            for idx in df.index
        ]
        return df

    def generate_failure_gallery(self, results: Dict, image: np.ndarray,
                                  top_n: int = 6) -> List[Dict]:
        """Identify worst failure cases."""
        failures = []
        baseline_class = results["baseline"]["predicted_class"]

        for corr_name, severities in results["corruptions"].items():
            for sev, result in severities.items():
                if result["class_changed"] or result["confidence"] < 0.3:
                    failures.append({
                        "corruption": corr_name,
                        "severity": sev,
                        "predicted_class": result["predicted_class"],
                        "confidence": result["confidence"],
                        "confidence_drop": results["baseline"]["confidence"] - result["confidence"],
                        "severity_name": CORRUPTIONS[corr_name].name if corr_name in CORRUPTIONS else corr_name,
                    })

        failures.sort(key=lambda x: x["confidence_drop"], reverse=True)
        return failures[:top_n]

    def generate_recommendations(self, results: Dict) -> List[Dict]:
        """Generate actionable recommendations — FIXED: context-aware, not panic mode."""

        recommendations = []
        metrics = results["metrics"]
        corruptions = results["corruptions"]
        phi = metrics["harmonic_mean"]
        risk = metrics["risk_score"]

        # =========================================================
        # OVERALL VERDICT (driven by Φ score, not individual corruptions)
        # =========================================================

        if phi >= 80:
            recommendations.append({
                "priority": "low",
                "title": "Model Deployment Ready",
                "description": (
                    f"Φ composite score of {phi:.1f}/100 indicates strong robustness. "
                    f"Risk score: {risk:.1f}/100. Model handles most corruptions well."
                ),
                "action": "Deploy with standard monitoring. No critical interventions needed."
            })

        elif phi >= 50:
            recommendations.append({
                "priority": "medium",
                "title": "Conditional Deployment",
                "description": (
                    f"Φ score of {phi:.1f}/100 shows moderate robustness. "
                    f"Some corruptions cause degradation but model remains usable."
                ),
                "action": "Deploy with confidence thresholding (≥60%) and input validation."
            })

        else:
            recommendations.append({
                "priority": "critical",
                "title": "High Risk — Do Not Deploy",
                "description": (
                    f"Φ score of {phi:.1f}/100 indicates poor robustness. "
                    f"Model fails systematically under real-world conditions."
                ),
                "action": "Do not deploy. Retrain with corruption augmentation first."
            })

        # =========================================================
        # ONLY flag critical corruptions if Φ is actually concerning
        # =========================================================

        if phi < 70:  # Only show corruption-level warnings if model isn't already strong
            corr_scores = {}
            for corr_name, severities in corruptions.items():
                avg_conf = np.mean([r["confidence"] for r in severities.values()])
                corr_scores[corr_name] = avg_conf

            worst_corruptions = sorted(corr_scores.items(), key=lambda x: x[1])

            for corr_name, avg_conf in worst_corruptions[:2]:  # Max 2, not 3
                corr_info = CORRUPTIONS[corr_name] if corr_name in CORRUPTIONS else None
                corr_label = corr_info.name if corr_info else corr_name

                if avg_conf < 0.3:
                    recommendations.append({
                        "priority": "high",
                        "title": f"Critical: {corr_label}",
                        "description": (
                            f"Confidence collapses to {avg_conf * 100:.1f}% under {corr_label}. "
                            f"This is a severe failure mode requiring immediate attention."
                        ),
                        "action": f"Add {corr_label.lower()} augmentation and consider ensemble methods."
                    })
                elif avg_conf < 0.5 and phi < 60:
                    recommendations.append({
                        "priority": "medium",
                        "title": f"Weakness: {corr_label}",
                        "description": (
                            f"Model shows sensitivity to {corr_label} "
                            f"(avg confidence: {avg_conf * 100:.1f}%)."
                        ),
                        "action": f"Increase {corr_label.lower()} diversity in training data."
                    })

        # =========================================================
        # STABILITY: only flag if actually bad
        # =========================================================

        if metrics["stability"] < 40:
            recommendations.append({
                "priority": "high",
                "title": "Low Prediction Stability",
                "description": (
                    f"Only {metrics['stability']:.1f}% prediction consistency under corruption. "
                    f"Model frequently changes its mind when stressed."
                ),
                "action": "Use test-time augmentation or ensemble voting for consistency."
            })

        # =========================================================
        # TOP-5: only flag if actually bad
        # =========================================================

        if metrics["top5_accuracy"] < 60 and phi < 70:
            recommendations.append({
                "priority": "medium",
                "title": "Weak Top-5 Robustness",
                "description": (
                    f"Correct class leaves top-5 in {100 - metrics['top5_accuracy']:.1f}% of cases."
                ),
                "action": "Use label smoothing and larger training datasets."
            })

        return recommendations

    def generate_severity_chart(self, results: Dict) -> Dict:
        """Generate severity degradation chart data."""
        corruptions = results["corruptions"]
        chart_data = {}
        for corr_name, severities in corruptions.items():
            corr_label = CORRUPTIONS[corr_name].name if corr_name in CORRUPTIONS else corr_name
            severity_values = {}
            for severity, result in severities.items():
                confidence = float(result["confidence"]) * 100
                severity_values[f"Severity {severity}"] = confidence
            chart_data[corr_label] = severity_values
        return chart_data

    def export_csv(self, results: Dict) -> str:
        """Export results to CSV format."""
        rows = []
        baseline = results["baseline"]
        for corr_name, severities in results["corruptions"].items():
            for sev, result in severities.items():
                rows.append({
                    "corruption_type": corr_name,
                    "severity": sev,
                    "baseline_class": baseline["predicted_class"],
                    "baseline_confidence": baseline["confidence"],
                    "corrupted_class": result["predicted_class"],
                    "corrupted_confidence": result["confidence"],
                    "class_changed": result["class_changed"],
                    "confidence_drop": baseline["confidence"] - result["confidence"],
                    "top5_retained": result["top5_retained"],
                })
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)

    def export_summary(self, results: Dict) -> str:
        """Generate text summary report."""
        metrics = results["metrics"]
        baseline = results["baseline"]

        report = f"""
========================================
PREFLIGHT AI - ROBUSTNESS REPORT
========================================

BASELINE PREDICTION
-------------------
Class: {baseline['predicted_class']}
Confidence: {baseline['confidence']*100:.2f}%

ROBUSTNESS METRICS
------------------
Prediction Stability: {metrics['stability']:.1f}%
Confidence Retention: {metrics['confidence_retention']:.1f}%
Top-5 Accuracy: {metrics['top5_accuracy']:.1f}%
Harmonic Mean (Φ): {metrics['harmonic_mean']:.1f}

RISK ASSESSMENT
---------------
Risk Score: {metrics['risk_score']:.1f}/100
Category: {self.compute_risk_category(metrics['risk_score']).upper()}

Total Tests Run: {metrics['total_tests']}

RECOMMENDATIONS
---------------
"""
        for rec in self.generate_recommendations(results):
            report += (
                f"\n[{rec['priority'].upper()}] {rec['title']}\n"
                f"  {rec['description']}\n"
                f"  Action: {rec['action']}\n"
            )
        return report