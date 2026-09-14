# 🚀 PreFlight AI

**Pre-Deployment Robustness Validation Platform for Computer Vision Models**

> *"Don't let your CV model fail in production. Test it before you ship it."*

---

## ✨ Features

### 🤖 Agent-Based Corruption Selection
- **8 Intelligent Agents** that analyze your images and auto-select the best corruption suite
- Agents: Night Vision 🌙, Action Camera 🏃, Surveillance 📹, Autonomous Vehicle 🚗, Mobile App 📱, Security AI 🔒, Medical 🏥, General 🔬
- Each agent prioritizes corruptions based on deployment scenario

### ⚔️ Real Adversarial Attacks
- **FGSM** (Fast Gradient Sign Method) — single-step gradient attack
- **PGD** (Projected Gradient Descent) — iterative gradient attack
- Uses **real backpropagation** through pretrained neural networks
- Visual before/after comparison showing prediction flips

### 📊 Comprehensive Metrics
- **Prediction Stability** — % of predictions that don't flip class
- **Confidence Retention** — average confidence maintained under corruption
- **Top-5 Accuracy** — % where correct class stays in top-5
- **Harmonic Mean** — mathematically principled composite score
- **Risk Score** (0-100) — deployment readiness indicator

### 🤖 Claude AI Analysis
- Optional Claude API integration for intelligent report generation
- Generates downloadable HTML reports with executive summaries
- Provides actionable improvement roadmaps
- Falls back to template analysis if API unavailable

### 📈 Visualizations
- **Vulnerability Heatmaps** — corruption × severity confidence matrix
- **Severity Degradation Charts** — interactive Plotly line charts
- **Failure Case Gallery** — worst-performing corruption examples
- **Adversarial Visualization** — side-by-side attack comparison

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

### 3. Optional: Configure Claude API

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

---

## 📁 Project Structure

```
preflight-ai/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration & constants
├── corruption_engine.py      # Image corruption implementations
├── model_predictor.py        # Neural network inference
├── agent_system.py           # Intelligent agent selection
├── robustness_evaluator.py   # Testing orchestration
├── metrics_analyzer.py       # Scoring & recommendations
├── claude_analyzer.py        # Claude API report generation
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── assets/                   # Sample images & resources
```

---

## 🎯 How It Works

```
1. UPLOAD IMAGES → Drag & drop test images
        ↓
2. AGENT SELECTION → AI analyzes images & picks best agent
        ↓
3. CORRUPTION TESTING → Applies 6-9 corruptions × 3-5 severities
        ↓
4. INFERENCE → Real ResNet50/MobileNet/EfficientNet forward passes
        ↓
5. SCORING → Harmonic mean of stability + confidence retention
        ↓
6. REPORTING → Heatmaps, charts, recommendations, Claude analysis
```

---

## 🔬 Corruption Categories

| Category | Corruptions | Description |
|----------|-------------|-------------|
| **Environmental** | Gaussian Noise, Motion Blur, Fog, Brightness | Real-world camera/sensor issues |
| **Digital** | JPEG Compression, Pixelation, Gaussian Blur | Transmission & storage artifacts |
| **Geometric** | Occlusion, Rotation | Partial obstruction & perspective |
| **Adversarial** | FGSM, PGD | Gradient-based security attacks |

---

## 📊 Scoring Methodology

The composite robustness score uses a **harmonic mean** to penalize models that achieve high stability through low-confidence predictions:

```
Φ = 2 × (S × R) / (S + R)

Where:
  S = Prediction Stability (%)
  R = Confidence Retention (%)

Risk Score = (1 - Φ/100) × 100
```

| Risk Score | Category | Action |
|------------|----------|--------|
| 0-30 | 🟢 LOW | Production ready |
| 31-60 | 🟡 MEDIUM | Needs targeted improvements |
| 61-100 | 🔴 HIGH | Not recommended for deployment |

---

## 🧠 Agent System

The agent system analyzes image features to select the optimal testing strategy:

- **Brightness & contrast** → Night Vision detection
- **Edge density & motion** → Action Camera detection
- **Saturation & color** → Mobile/Surveillance detection
- **Aspect ratio & objects** → Autonomous Vehicle detection

Each agent weights corruptions differently — e.g., the Surveillance agent prioritizes JPEG compression and pixelation, while the Autonomous agent prioritizes fog and motion blur.

---

## 📝 Patent Context

This is a **patent demonstration prototype** for academic evaluation, showcasing:

1. **Novel approach** to pre-deployment CV model testing
2. **Real AI/ML implementation** — not hardcoded logic
3. **Clear differentiation** from Bosch 2026 industrial virtual testing
4. **Web-native accessibility** — no specialized hardware needed

### Key Differentiators vs. Bosch 2026
- **Domain**: General software/web apps vs. industrial manufacturing
- **Architecture**: Web-native SaaS vs. on-premise enterprise
- **Users**: Any developer vs. specialized factory engineers
- **Scope**: Multi-domain vs. manufacturing-only

---

## ⚠️ Limitations

- Evaluates single-image robustness (not video sequences)
- Corruption library doesn't exhaust all environmental conditions
- Scoring weights stability and confidence equally
- Requires PyTorch models (ONNX support planned)
- Claude API requires separate API key

---

## 📜 License

Academic/Patent Demonstration — Not for production use without further development.

---

**Built with ❤️ for the PreFlight AI Patent Demonstration**
