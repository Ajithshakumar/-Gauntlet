"""
PreFlight AI - Streamlit Application (FIXED)
=============================================
Clean pipeline: Upload → User picks scenario → Systematic Test → Score → Report.
Agent is USER-SELECTED, not fake-detected. Bar chart hidden when detection disabled.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go

import streamlit as st

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================

st.set_page_config(
    page_title="PreFlight AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# IMPORT PROJECT MODULES
# =============================================================================

try:
    from config import CORRUPTIONS, AGENTS, CORE_CORRUPTION_KEYS
    from corruption_engine import CorruptionEngine
    from model_predictor import ModelPredictor
    from agent_system import AgentSelector
    from robustness_evaluator import RobustnessEvaluator
    from metrics_analyzer import MetricsAnalyzer
    from api_analyzer import ClaudeAnalyzer
except Exception as e:
    st.error(f"Import Error: {e}")
    st.stop()

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 10px;
}
.sub-header {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
.metric-card {
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
}
.risk-low { background: #16a34a; }
.risk-medium { background: #f59e0b; }
.risk-high { background: #dc2626; }
.agent-card {
    background: #1f2937;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #6366f1;
    color: white;
}
.recommendation {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.recommendation.high { background: #fee2e2; }
.recommendation.medium { background: #fef3c7; }
.recommendation.low { background: #dcfce7; }
.patent-note {
    background: #1e3a5f;
    padding: 12px;
    border-radius: 8px;
    color: #93c5fd;
    font-size: 0.9rem;
    margin-bottom: 15px;
}
.scenario-select {
    background: #111827;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #374151;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

defaults = {
    "results": None,
    "images": [],
    "agent_info": None,
    "selected_model": "resnet50",
    "max_severity": 3,
    "use_ai_report": False,
    "ai_report_text": None,
    "user_agent_choice": "general",
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    model_options = {
        "resnet50": "ResNet-50 (80.86%)",
        "resnet18": "ResNet-18 (69.76%)",
        "mobilenet_v2": "MobileNet-V2 (72.15%)",
        "efficientnet_b0": "EfficientNet-B0 (77.69%)",
    }

    st.session_state.selected_model = st.selectbox(
        "Model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )

    st.session_state.max_severity = st.slider(
        "Max Severity Level",
        1, 3, 3,
        help="Patent-aligned: 3 severity levels per corruption category"
    )

    st.markdown("---")
    st.info("""
    **Pipeline:** Upload → Pick Scenario → Test All 18 → Score → Report
    """)

# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    '<div class="main-header">🚀 PreFlight AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Pre-Deployment Robustness Validation Platform</div>',
    unsafe_allow_html=True
)

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload",
    "🎯 Scenario",
    "📊 Results",
    "📋 Report"
])

# =============================================================================
# TAB 1 - UPLOAD
# =============================================================================

with tab1:
    st.header("Upload Test Images")

    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.images = []
        cols = st.columns(min(4, len(uploaded_files)))

        for i, file in enumerate(uploaded_files):
            try:
                image = Image.open(file).convert("RGB")
                arr = np.array(image)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                st.session_state.images.append(bgr)

                with cols[i % 4]:
                    st.image(image, caption=f"Image {i+1}", use_container_width=True)
            except Exception as e:
                st.error(f"Image load failed: {e}")

        st.success(f"Loaded {len(st.session_state.images)} images")

        if st.button("🔍 Baseline Prediction"):
            try:
                predictor = ModelPredictor(st.session_state.selected_model)
                for i, img in enumerate(st.session_state.images[:4]):
                    pred = predictor.predict(img)
                    st.write(
                        f"**Image {i+1}:** {pred['predicted_class']} "
                        f"({pred['confidence']*100:.1f}%)"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.session_state.images = []
        st.session_state.results = None

# =============================================================================
# TAB 2 - USER-SELECTED SCENARIO + VALIDATION
# =============================================================================

with tab2:
    st.header("🎯 Select Deployment Scenario")

    if not st.session_state.images:
        st.warning("Upload images in Tab 1 first.")
    else:
        try:
            selector = AgentSelector()
            all_agents = selector.get_all_agents()

            # USER SELECTS SCENARIO
            agent_options = {a["id"]: f"{a['icon']} {a['name']}" for a in all_agents}

            st.markdown("<div class='scenario-select'>", unsafe_allow_html=True)
            selected_agent_id = st.selectbox(
                "Choose your deployment scenario (optional):",
                list(agent_options.keys()),
                format_func=lambda x: agent_options[x],
                index=list(agent_options.keys()).index("general")
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.session_state.user_agent_choice = selected_agent_id

            # Get agent info with user choice
            sample_img = st.session_state.images[0]
            agent, reasoning = selector.select_agent(sample_img, user_preference=selected_agent_id)
            st.session_state.agent_info = reasoning

            st.markdown(f"""
            <div class="agent-card">
                <h3>{agent.icon} {agent.name}</h3>
                <p>{agent.description}</p>
                <p><strong>Testing Profile:</strong> {agent.reasoning}</p>
            </div>
            """, unsafe_allow_html=True)

            # HIDE BOGUS BAR CHART — show a simple tag cloud instead
            st.markdown("### Scenario Tags")
            tags = agent.trigger_conditions[:5]
            tag_html = " ".join([f'<span style="background:#374151; padding:4px 10px; border-radius:12px; font-size:0.85rem; margin-right:6px;">{t}</span>' for t in tags])
            st.markdown(tag_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Scenario setup failed: {e}")

        st.markdown("---")

        st.markdown("""
        <div class="patent-note">
        🛡️ <strong>Systematic Validation:</strong> All 6 corruption categories 
        × 3 severity levels = <strong>18 tests per image</strong>. Scenario selection 
        is for organizational tagging only — every deployment gets the full corruption battery.
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 RUN FULL VALIDATION", use_container_width=True):
            try:
                evaluator = RobustnessEvaluator(st.session_state.selected_model)
                all_results = []
                progress = st.progress(0)

                for idx, img in enumerate(st.session_state.images):
                    result = evaluator.evaluate_image(
                        img,
                        severities=list(range(1, st.session_state.max_severity + 1))
                    )
                    all_results.append(result)
                    progress.progress((idx + 1) / len(st.session_state.images))

                st.session_state.results = all_results
                st.session_state.ai_report_text = None
                st.success(f"✅ Validation complete! {len(all_results)} image(s) tested across 18 corruption variants.")

            except Exception as e:
                st.error(f"Validation failed: {e}")

# =============================================================================
# TAB 3 - RESULTS
# =============================================================================

with tab3:
    if st.session_state.results is None:
        st.info("Run validation in Tab 2 first.")
    else:
        try:
            results = st.session_state.results[0]
            metrics = results["metrics"]
            risk_score = metrics["risk_score"]

            if risk_score <= 30:
                risk_class = "risk-low"
                risk_text = "LOW RISK — PRODUCTION READY"
            elif risk_score <= 60:
                risk_class = "risk-medium"
                risk_text = "MEDIUM RISK — CONDITIONAL DEPLOY"
            else:
                risk_class = "risk-high"
                risk_text = "HIGH RISK — DO NOT DEPLOY"

            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <h1>{metrics['harmonic_mean']:.1f}/100</h1>
                <h3>Φ Composite Score</h3>
                <p style="font-size:1.1rem; margin-top:8px;">{risk_text}</p>
                <p style="opacity:0.8;">Risk Score: {risk_score:.1f}/100</p>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stability (S)", f"{metrics['stability']:.1f}%")
            c2.metric("Confidence Retention (R)", f"{metrics['confidence_retention']:.1f}%")
            c3.metric("Top-5 Accuracy", f"{metrics['top5_accuracy']:.1f}%")
            c4.metric("Tests Run", f"{metrics['total_tests']}")

            st.markdown("---")

            # VULNERABILITY HEATMAP
            st.subheader("🔥 Vulnerability Heatmap")
            analyzer = MetricsAnalyzer()
            heatmap_data = analyzer.generate_heatmap_data(results)

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale="RdYlGn_r",
                text=heatmap_data.values,
                texttemplate="%{text:.1f}",
                hoverongaps=False
            ))
            fig_heatmap.update_layout(
                title="Confidence Retention Across Corruptions & Severities",
                paper_bgcolor="#0B1020",
                plot_bgcolor="#111827",
                font_color="white",
                height=500
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.markdown("---")

            # SEVERITY DEGRADATION
            st.subheader("📉 Severity Degradation Curves")
            chart_data = analyzer.generate_severity_chart(results)
            chart_df = pd.DataFrame(chart_data).T

            fig2 = go.Figure()
            for column in chart_df.columns:
                fig2.add_trace(go.Scatter(
                    x=chart_df.index,
                    y=chart_df[column],
                    mode="lines+markers",
                    name=column
                ))
            fig2.update_layout(
                title="Confidence (%) vs Corruption Type",
                paper_bgcolor="#0B1020",
                plot_bgcolor="#111827",
                font_color="white",
                height=500,
                xaxis_title="Corruption Type",
                yaxis_title="Confidence (%)"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")

            # RECOMMENDATIONS
            st.subheader("💡 Recommendations")
            recommendations = analyzer.generate_recommendations(results)
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation {rec['priority']}">
                    <strong>[{rec['priority'].upper()}] {rec['title']}</strong><br>
                    {rec['description']}<br><br>
                    <em>→ {rec['action']}</em>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Results rendering failed: {e}")

# =============================================================================
# TAB 4 - REPORT
# =============================================================================

with tab4:
    if st.session_state.results is None:
        st.info("Run validation in Tab 2 first.")
    else:
        results = st.session_state.results[0]
        agent_info = st.session_state.agent_info or {}
        analyzer = MetricsAnalyzer()

        st.subheader("📄 Export Reports")

        # Text summary (always available)
        summary_text = analyzer.export_summary(results)
        st.download_button(
            label="📥 Download Text Summary",
            data=summary_text,
            file_name="preflight_summary.txt",
            mime="text/plain"
        )

        # CSV export
        csv_data = analyzer.export_csv(results)
        st.download_button(
            label="📥 Download CSV Data",
            data=csv_data,
            file_name="preflight_data.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # AI Report (on-demand only)
        st.subheader("🤖 AI Deployment Analysis")

        if st.button("Generate AI Analysis (xAI/Gemini)"):
            with st.spinner("Generating intelligent analysis..."):
                try:
                    claude = ClaudeAnalyzer()
                    report = claude.analyze(results, agent_info)
                    st.session_state.ai_report_text = report
                except Exception as e:
                    st.error(f"AI generation failed: {e}")

        if st.session_state.ai_report_text:
            st.markdown(st.session_state.ai_report_text)

            # HTML report download
            html_report = ClaudeAnalyzer().generate_html_report(
                results, agent_info, st.session_state.ai_report_text
            )
            st.download_button(
                label="📥 Download Professional HTML Report",
                data=html_report,
                file_name="preflight_ai_report.html",
                mime="text/html"
            )

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray;">
    <p>🚀 <strong>PreFlight AI</strong> — Pre-Deployment Robustness Validation</p>
    <p style="font-size:0.8rem;">Systematic 6×3 corruption testing with composite Φ scoring</p>
</div>
""", unsafe_allow_html=True)