"""
PreFlight AI - API Analyzer (XAI PRIMARY + GEMINI FALLBACK)
============================================================
Primary: xAI Grok API via OpenAI-compatible endpoint.
Fallback: Google Gemini API if xAI fails or key missing.
Final fallback: Hardcoded professional template.
"""

import os
from typing import Dict, Optional


class ClaudeAnalyzer:
    """Uses xAI Grok primary, Gemini fallback, template final fallback."""

    def __init__(self, xai_key: Optional[str] = None, gemini_key: Optional[str] = None):
        self.xai_key = xai_key or os.getenv("XAI_API_KEY")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        self.xai_model = os.getenv("XAI_MODEL", "grok-2-latest")
        self.gemini_model = "gemini-1.5-flash"
        self.xai_base_url = "https://api.x.ai/v1"

    def _build_prompt(self, results: Dict, agent_info: Dict) -> str:
        """Build focused prompt — shared across both LLMs."""
        metrics = results["metrics"]
        baseline = results["baseline"]
        corruptions = results["corruptions"]

        worst = []
        for corr_name, severities in corruptions.items():
            avg_conf = sum(r["confidence"] for r in severities.values()) / len(severities)
            worst.append((corr_name, avg_conf))
        worst.sort(key=lambda x: x[1])

        worst_summary = "\n".join([
            f"- {name}: avg confidence {conf*100:.1f}%"
            for name, conf in worst[:3]
        ])

        return f"""You are a senior MLOps engineer writing a pre-deployment readiness memo.

MODEL: {baseline['predicted_class']} ({baseline['confidence']*100:.1f}% baseline confidence)
SCORES: Φ={metrics['harmonic_mean']:.1f}/100 | S={metrics['stability']:.1f}% | R={metrics['confidence_retention']:.1f}% | Risk={metrics['risk_score']:.1f}/100
WEAKEST CORRUPTIONS:
{worst_summary}

Write EXACTLY 4 bullet points:
1. The single biggest failure mode and why it matters
2. One specific data augmentation to fix it
3. A deployment safeguard (e.g., confidence threshold)
4. Whether this model is safe for safety-critical use (yes/no/maybe + one sentence)

Be specific. No generic advice. No introductions. Bullet points only."""

    def _call_xai(self, prompt: str) -> Optional[str]:
        """Call xAI Grok. Returns text or None on failure."""
        if not self.xai_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.xai_key, base_url=self.xai_base_url)
            response = client.chat.completions.create(
                model=self.xai_model,
                messages=[
                    {"role": "system", "content": "You are an expert MLOps engineer. Be concise and specific."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception:
            return None

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Google Gemini. Returns text or None on failure."""
        if not self.gemini_key:
            return None

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel(self.gemini_model)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            return None

    def analyze(self, results: Dict, agent_info: Dict) -> str:
        """Try xAI → Gemini → Template."""
        prompt = self._build_prompt(results, agent_info)

        # 1. Try xAI (primary)
        text = self._call_xai(prompt)
        if text:
            return f"<!-- Powered by xAI Grok -->\n{text}"

        # 2. Try Gemini (fallback)
        text = self._call_gemini(prompt)
        if text:
            return f"<!-- Powered by Google Gemini (xAI fallback) -->\n{text}"

        # 3. Hardcoded template (final fallback)
        return self._fallback_analysis(results, agent_info)

    def _fallback_analysis(self, results: Dict, agent_info: Dict) -> str:
        """Hardcoded template — looks professional even without any API."""
        metrics = results["metrics"]
        baseline = results["baseline"]
        risk_score = metrics["risk_score"]

        if risk_score <= 30:
            verdict, emoji = "PRODUCTION READY", "✅"
        elif risk_score <= 60:
            verdict, emoji = "CONDITIONAL DEPLOYMENT", "⚠️"
        else:
            verdict, emoji = "DEPLOYMENT NOT RECOMMENDED", "❌"

        worst = []
        for corr_name, severities in results["corruptions"].items():
            avg_conf = sum(r["confidence"] for r in severities.values()) / len(severities)
            worst.append((corr_name, avg_conf))
        worst.sort(key=lambda x: x[1])

        report = f"""## {emoji} PreFlight AI Readiness Report

| Metric | Value |
|--------|-------|
| **Baseline Class** | {baseline['predicted_class']} |
| **Baseline Confidence** | {baseline['confidence']*100:.1f}% |
| **Φ Composite Score** | {metrics['harmonic_mean']:.1f}/100 |
| **Stability (S)** | {metrics['stability']:.1f}% |
| **Confidence Retention (R)** | {metrics['confidence_retention']:.1f}% |
| **Top-5 Accuracy** | {metrics['top5_accuracy']:.1f}% |
| **Risk Score** | {metrics['risk_score']:.1f}/100 |

### Verdict: {verdict}

---

### Critical Vulnerabilities

"""
        for i, (corr_name, avg_conf) in enumerate(worst[:3], 1):
            sev = "🔴 CRITICAL" if avg_conf < 0.3 else "🟠 HIGH" if avg_conf < 0.5 else "🟡 MEDIUM"
            report += f"{i}. **{corr_name.upper()}** {sev} — avg confidence drops to {avg_conf*100:.1f}%\n"

        report += f"""
---

### Deployment Checklist

- [ ] **Confidence Thresholding**: Reject predictions below 60% confidence in production
- [ ] **Input Validation**: Block compressed or low-light images before inference
- [ ] **Human-in-the-Loop**: Required for {baseline['predicted_class']} classifications when risk > 60
- [ ] **Monitoring**: Track prediction stability drift weekly

---

### Improvement Actions

1. **Immediate**: Add `{worst[0][0]}` augmentation to training pipeline
2. **Short-term**: Retrain with corrupted dataset (recommended: ImgAug or Albumentations)
3. **Long-term**: Implement test-time augmentation (TTA) for critical inputs

---

*Report generated by PreFlight AI | Scenario: {agent_info.get('name', 'General')}*
"""
        return report

    def generate_html_report(self, results: Dict, agent_info: Dict,
                             analysis_text: str = None) -> str:
        """Generate full HTML report."""
        if analysis_text is None:
            analysis_text = self.analyze(results, agent_info)

        metrics = results["metrics"]
        baseline = results["baseline"]
        risk_score = metrics["risk_score"]

        if risk_score <= 30:
            risk_color, verdict = "#27ae60", "PRODUCTION READY"
        elif risk_score <= 60:
            risk_color, verdict = "#f39c12", "CONDITIONAL"
        else:
            risk_color, verdict = "#e74c3c", "NOT RECOMMENDED"

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>PreFlight AI Report</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0B1020; color: #e2e8f0; padding: 40px; }}
.container {{ max-width: 1000px; margin: auto; }}
.banner {{ background: {risk_color}; padding: 24px; border-radius: 12px; margin-bottom: 30px; text-align: center; font-size: 1.3rem; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }}
.card {{ background: #111827; padding: 20px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); text-align: center; }}
.value {{ font-size: 2.2rem; font-weight: bold; color: {risk_color}; }}
.analysis {{ background: #111827; padding: 28px; border-radius: 14px; line-height: 1.7; }}
h1, h2, h3 {{ color: white; margin-top: 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
th {{ color: #94a3b8; font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
<h1>🚀 PreFlight AI Robustness Report</h1>
<div class="banner">
    <strong>Φ Score: {metrics['harmonic_mean']:.1f}/100</strong> — {verdict}
    <div style="font-size:0.9rem; margin-top:6px; opacity:0.9;">Risk Score: {risk_score:.1f}/100</div>
</div>
<div class="metrics">
    <div class="card"><div class="value">{metrics['stability']:.1f}%</div><div>Stability</div></div>
    <div class="card"><div class="value">{metrics['confidence_retention']:.1f}%</div><div>Retention</div></div>
    <div class="card"><div class="value">{metrics['top5_accuracy']:.1f}%</div><div>Top-5 Accuracy</div></div>
</div>
<div class="metrics">
    <div class="card"><div class="value">{baseline['predicted_class']}</div><div>Baseline Class</div></div>
    <div class="card"><div class="value">{baseline['confidence']*100:.1f}%</div><div>Baseline Confidence</div></div>
    <div class="card"><div class="value">{metrics['total_tests']}</div><div>Tests Run</div></div>
</div>
<div class="analysis">
{analysis_text.replace(chr(10), '<br>')}
</div>
</div>
</body>
</html>"""
        return html