"""
Smart Waste Classification System - Streamlit Web Application
An interactive academic dashboard for deep learning-based waste categorization.
"""

import os
import sys
from pathlib import Path
import json

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import config
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.prediction import predict_waste, load_model_cached
from training.preprocess import explore_dataset


# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Smart Waste Classification | AI for Sustainability",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Google Fonts & Root Variables */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --primary-green: #2e7d32;
        --primary-dark: #1b5e20;
        --primary-light: #4caf50;
        --mint-bg: #f4fbf7;
        --card-bg: #ffffff;
        --text-main: #1f2937;
        --text-muted: #4b5563;
        --border-soft: #e5e7eb;
        --shadow-sm: 0 2px 8px rgba(46, 125, 50, 0.08);
        --shadow-md: 0 6px 18px rgba(46, 125, 50, 0.12);
    }

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Styling */
    .stApp {
        background-color: var(--mint-bg);
        color: var(--text-main);
    }

    /* Eco Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #43a047 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(27, 94, 32, 0.2);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 400;
        opacity: 0.95;
        line-height: 1.5;
        max-width: 850px;
    }

    /* Metric & Feature Cards */
    .stat-card {
        background: var(--card-bg);
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    .stat-num {
        font-size: 2rem;
        font-weight: 800;
        color: var(--primary-green);
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }

    /* Category Cards */
    .cat-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.4rem;
        border: 1px solid var(--border-soft);
        box-shadow: var(--shadow-sm);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .cat-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .cat-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-main);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }

    /* Prediction Result Cards */
    .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.8rem;
        border: 2px solid #81c784;
        box-shadow: var(--shadow-md);
        margin-bottom: 1.5rem;
    }
    .result-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 50px;
        background: #e8f5e9;
        color: #2e7d32;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Warning Card */
    .alert-card {
        background: #fff8e1;
        border-left: 5px solid #ffa000;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        color: #5d4037;
        margin: 1rem 0;
    }

    /* Step Pipeline Item */
    .step-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }
    .step-icon {
        font-size: 2rem;
        margin-bottom: 0.4rem;
    }
    .step-title {
        font-weight: 700;
        color: var(--primary-dark);
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }
    .step-desc {
        font-size: 0.82rem;
        color: var(--text-muted);
    }

    /* Sidebar Clean Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3.2rem;">🌱</div>
            <h2 style="color: #1b5e20; margin: 0.2rem 0; font-weight: 800; font-size: 1.5rem;">EcoSort AI</h2>
            <p style="color: #6b7280; font-size: 0.85rem; margin: 0;">Smart Waste Classification</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    selected_page = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Overview",
            "🔍 Image Classification",
            "📊 Model Performance",
            "ℹ️ About the Project",
        ],
        index=0,
    )

    st.divider()

    st.markdown("### ⚙️ Inference Engine")
    active_model_key = st.selectbox(
        "Default Model Architecture",
        options=["mobilenet", "cnn"],
        format_func=lambda x: "MobileNetV2 (Transfer Learning)" if x == "mobilenet" else "Custom Deep CNN",
        index=0,
        help="Choose which trained neural network performs image inference.",
    )

    st.markdown(
        """
        <div style="margin-top: 1.5rem; padding: 0.85rem; background: #e8f5e9; border-radius: 10px; font-size: 0.8rem; color: #1b5e20;">
            <b>Academic Mini-Project</b><br>
            B.E. Computer Engineering<br>
            Deep Learning & Computer Vision
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# PAGE 1: HOME OVERVIEW
# ==========================================
if selected_page == "🏠 Home Overview":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">♻️ Smart Waste Classification System</div>
            <div class="hero-subtitle">
                An end-to-end Computer Vision platform powered by Deep Convolutional Neural Networks and Transfer Learning.
                Automatically identifies recyclable and non-recyclable solid municipal waste to accelerate sustainable sorting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Stat Callouts
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-num">2,527</div>
                <div class="stat-label">TrashNet Dataset Images</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-num">6</div>
                <div class="stat-label">Material Categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-num">2</div>
                <div class="stat-label">DL Architectures (CNN & MobileNet)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-num">70:15:15</div>
                <div class="stat-label">Train-Val-Test Stratified Split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔄 How the Classification System Works")
    st.markdown("Our automated pipeline guides waste from raw image capture to actionable recycling intelligence:")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-icon">📸</div>
                <div class="step-title">1. Image Ingestion</div>
                <div class="step-desc">Accepts user uploads or live camera streams, resizing and validating inputs to 224×224 pixels.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-icon">🧠</div>
                <div class="step-title">2. Deep Feature Extraction</div>
                <div class="step-desc">Extracts multi-scale spatial textures, edges, and materials using CNN kernels or MobileNetV2 inverted residuals.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-icon">🎯</div>
                <div class="step-title">3. Softmax Classification</div>
                <div class="step-desc">Computes normalized probability distributions over all 6 waste classes with uncertainty bounds.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-icon">♻️</div>
                <div class="step-title">4. Eco Action Protocol</div>
                <div class="step-desc">Surfaces disposal protocols, contamination warnings, and carbon offset statistics.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📦 Recognized Waste Categories & Disposal Directives")

    # Render 6 category cards in 2 rows of 3 columns
    cols_row1 = st.columns(3)
    cols_row2 = st.columns(3)
    cards = list(config.CATEGORY_INFO.items())

    for idx, (cls_name, info) in enumerate(cards):
        col = cols_row1[idx] if idx < 3 else cols_row2[idx - 3]
        with col:
            st.markdown(
                f"""
                <div class="cat-card" style="border-top: 4px solid {info['color']}; margin-bottom: 1.2rem;">
                    <div>
                        <span class="cat-badge" style="background: {info['bg_color']}; color: {info['color']};">
                            {info['badge']}
                        </span>
                        <div class="cat-title">{info['icon']} {info['title']}</div>
                        <p style="color: #4b5563; font-size: 0.88rem; margin-bottom: 0.8rem;">
                            {info['description']}
                        </p>
                    </div>
                    <div style="background: #fafafa; border-radius: 8px; padding: 0.65rem; font-size: 0.8rem; border: 1px solid #eee;">
                        <b style="color: {info['color']};">💡 Tip:</b> {info['recycling_tip']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==========================================
# PAGE 2: IMAGE CLASSIFICATION
# ==========================================
elif selected_page == "🔍 Image Classification":
    st.markdown("## 🔍 Live Waste Classification")
    st.markdown(
        "Upload a photograph of a waste item or select a curated sample from our independent test dataset. "
        "The model analyzes optical patterns to determine material category, prediction confidence, and disposal guidance."
    )

    # Input method selection
    tab_upload, tab_samples = st.tabs(["📤 Upload Image", "🖼️ Curated Test Samples"])

    image_to_classify = None
    image_source_label = ""

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a waste image (JPG, JPEG, or PNG)",
            type=["jpg", "jpeg", "png"],
            help="High-contrast images with centered waste items produce optimal classification.",
        )
        if uploaded_file is not None:
            image_to_classify = uploaded_file
            image_source_label = f"Uploaded File: {uploaded_file.name}"

    with tab_samples:
        st.markdown("Select a sample image from the test set:")
        # Look for sample test images in data/split/test
        sample_options = {}
        test_dir = config.TEST_DIR

        if test_dir.exists():
            for c in config.CLASSES:
                c_dir = test_dir / c
                if c_dir.exists():
                    images = list(c_dir.glob("*.*"))
                    if images:
                        sample_options[f"Sample {c.capitalize()} ({images[0].name})"] = images[0]

        if sample_options:
            selected_sample_key = st.selectbox(
                "Choose a sample test image:",
                options=list(sample_options.keys()),
                index=0,
            )
            if st.button("Load Selected Sample"):
                image_to_classify = sample_options[selected_sample_key]
                image_source_label = selected_sample_key
        else:
            st.info("Test split directory not yet populated. Please run training/preprocess.py first.")

    st.divider()

    # Classification Execution
    if image_to_classify is not None:
        col_img, col_pred = st.columns([1, 1.2])

        with col_img:
            st.markdown("#### 📷 Target Image")
            try:
                pil_img = Image.open(image_to_classify)
                st.image(pil_img, use_container_width=True, caption=image_source_label)

                # Metadata card
                w, h = pil_img.size
                st.markdown(
                    f"""
                    <div style="background: #ffffff; padding: 0.8rem 1rem; border-radius: 10px; border: 1px solid #e0e0e0; font-size: 0.85rem; color: #4b5563;">
                        <b>Format:</b> {pil_img.format or 'JPEG'} &nbsp;|&nbsp;
                        <b>Dimensions:</b> {w} × {h} px &nbsp;|&nbsp;
                        <b>Color Mode:</b> {pil_img.mode}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Error opening image: {e}")
                pil_img = None

        with col_pred:
            st.markdown("#### 🎯 Classification Result")
            if pil_img is not None:
                # Check model availability
                target_model_path = config.MOBILENET_MODEL_PATH if active_model_key == "mobilenet" else config.CNN_MODEL_PATH
                if not target_model_path.exists():
                    st.warning(
                        f"⚠️ The model weights for **{active_model_key.upper()}** are not found at `{target_model_path}`. "
                        "The model is currently being trained or has not yet completed. "
                        "Please check training status or wait for training to finish."
                    )
                else:
                    with st.spinner("Analyzing image features..."):
                        try:
                            result = predict_waste(
                                image_input=pil_img,
                                model_type=active_model_key,
                                confidence_threshold=0.45,
                            )

                            pred_cls = result["predicted_class"]
                            conf = result["confidence"]
                            info = result["category_info"]

                            # Top Result Card
                            st.markdown(
                                f"""
                                <div class="result-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                                        <span class="result-badge">
                                            {info.get('badge', 'Classified Material')}
                                        </span>
                                        <span style="font-size: 0.85rem; color: #6b7280;">
                                            Engine: <b>{result['model_used']}</b>
                                        </span>
                                    </div>
                                    <div style="font-size: 2.2rem; font-weight: 800; color: {info.get('color', '#2e7d32')};">
                                        {info.get('icon', '♻️')} {info.get('title', pred_cls.capitalize())}
                                    </div>
                                    <div style="font-size: 1.15rem; font-weight: 600; color: #374151; margin-top: 0.3rem;">
                                        Prediction Confidence: <b>{conf:.1%}</b>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Uncertainty Alert
                            if result["is_uncertain"]:
                                st.markdown(
                                    f"""
                                    <div class="alert-card">
                                        <b>⚠️ Low Confidence / Ambiguous Material Notice:</b><br>
                                        {result['warning_message']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            # Class-wise probability chart
                            st.markdown("##### 📊 Class-wise Probability Distribution")
                            prob_df = pd.DataFrame([
                                {"Category": c.capitalize(), "Probability": p}
                                for c, p in result["probabilities"].items()
                            ]).sort_values(by="Probability", ascending=True)

                            fig = px.bar(
                                prob_df,
                                x="Probability",
                                y="Category",
                                orientation="h",
                                text=prob_df["Probability"].apply(lambda v: f"{v:.1%}"),
                                color="Probability",
                                color_continuous_scale="Greens",
                                range_x=[0, 1.05],
                            )
                            fig.update_layout(
                                height=280,
                                margin=dict(l=0, r=20, t=20, b=20),
                                coloraxis_showscale=False,
                                xaxis_title=None,
                                yaxis_title=None,
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                            )
                            fig.update_traces(textposition="outside")
                            st.plotly_chart(fig, use_container_width=True)

                            # Actionable Guidance Cards
                            st.markdown("##### 💡 Disposal & Environmental Directive")
                            g1, g2 = st.columns(2)
                            with g1:
                                st.info(f"**Recycling Protocol:**\n\n{info.get('recycling_tip', 'Dispose according to local recycling ordinances.')}")
                            with g2:
                                st.success(f"**Ecological Impact:**\n\n{info.get('environmental_impact', 'Helps preserve natural resources.')}")

                        except Exception as e:
                            st.error(f"Inference error: {e}")
    else:
        st.info("👆 Please upload an image or choose a test sample from the tabs above to begin classification.")


# ==========================================
# PAGE 3: MODEL PERFORMANCE & BENCHMARKING
# ==========================================
elif selected_page == "📊 Model Performance":
    st.markdown("## 📊 Comprehensive Model Benchmarking")
    st.markdown(
        "Performance evaluation of the Custom CNN vs MobileNetV2 (Transfer Learning) on the independent test split (380 unseen images). "
        "All metrics and curves are generated directly from true model predictions."
    )

    # Check for comparison metrics file
    comp_file = config.COMPARISON_TABLE_PATH
    if comp_file.exists():
        with open(comp_file, "r") as f:
            comp_data = json.load(f)
        comp_df = pd.DataFrame(comp_data)

        # High-level KPIs
        k1, k2, k3, k4 = st.columns(4)
        cnn_row = comp_df[comp_df["Model"].str.contains("CNN")]
        mob_row = comp_df[comp_df["Model"].str.contains("MobileNet")]

        with k1:
            cnn_acc = cnn_row["Accuracy"].values[0] if not cnn_row.empty else 0.0
            st.metric("Custom CNN Test Accuracy", f"{cnn_acc:.2%}")
        with k2:
            mob_acc = mob_row["Accuracy"].values[0] if not mob_row.empty else 0.0
            st.metric("MobileNetV2 Test Accuracy", f"{mob_acc:.2%}", delta=f"{(mob_acc - cnn_acc)*100:+.1f}% vs CNN" if cnn_acc > 0 else None)
        with k3:
            mob_f1 = mob_row["Macro F1-Score"].values[0] if not mob_row.empty else 0.0
            st.metric("MobileNetV2 Macro F1", f"{mob_f1:.2%}")
        with k4:
            st.metric("Independent Test Samples", "380 Images")

        st.markdown("<br>", unsafe_allow_html=True)

        # Performance Comparison Table
        st.markdown("### 📋 Comparative Evaluation Matrix")
        display_df = comp_df.copy()
        for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score", "Weighted F1-Score"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda v: f"{v:.2%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Multi-Metric Comparison Chart
        comp_chart_path = config.GRAPHS_DIR / "model_comparison_chart.png"
        if comp_chart_path.exists():
            st.image(str(comp_chart_path), caption="Side-by-Side Model Evaluation Metrics", use_container_width=True)

    else:
        st.warning("Model comparison data not found yet. The models may still be training or evaluation has not been executed.")

    st.divider()

    # Detailed Inspection Tabs
    tab_cm, tab_curves, tab_dataset = st.tabs(["🔲 Confusion Matrices", "📈 Training & Loss Curves", "📁 Dataset Distribution"])

    with tab_cm:
        st.markdown("#### Confusion Matrix Breakdown")
        cm_model_choice = st.radio("Select Model for Confusion Matrix:", ["MobileNetV2 (Transfer Learning)", "Custom CNN"], horizontal=True)
        prefix = "mobilenet" if "MobileNet" in cm_model_choice else "cnn"

        c1, c2 = st.columns(2)
        norm_cm = config.CM_DIR / f"{prefix}_normalized.png"
        raw_cm = config.CM_DIR / f"{prefix}_raw.png"

        with c1:
            if norm_cm.exists():
                st.image(str(norm_cm), caption=f"{cm_model_choice} - Normalized Confusion Matrix", use_container_width=True)
            else:
                st.info("Normalized confusion matrix plot will appear once evaluation completes.")
        with c2:
            if raw_cm.exists():
                st.image(str(raw_cm), caption=f"{cm_model_choice} - Raw Counts Confusion Matrix", use_container_width=True)
            else:
                st.info("Raw counts confusion matrix plot will appear once evaluation completes.")

        # Class-wise classification report metrics
        report_file = config.METRICS_DIR / f"{prefix}_report.json"
        if report_file.exists():
            with open(report_file, "r") as f:
                rep_data = json.load(f)
            clf_rep = rep_data.get("classification_report", {})
            rows = []
            for cls in config.CLASSES:
                if cls in clf_rep:
                    rows.append({
                        "Category": cls.capitalize(),
                        "Precision": f"{clf_rep[cls]['precision']:.2%}",
                        "Recall": f"{clf_rep[cls]['recall']:.2%}",
                        "F1-Score": f"{clf_rep[cls]['f1-score']:.2%}",
                        "Support (Test Count)": int(clf_rep[cls]['support']),
                    })
            if rows:
                st.markdown(f"##### {cm_model_choice} Per-Class Performance Breakdown")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_curves:
        st.markdown("#### Training & Validation Trajectories")
        curve_choice = st.radio("Select Model Curves:", ["Custom CNN", "MobileNetV2 (Transfer Learning)"], horizontal=True)
        curve_path = config.GRAPHS_DIR / ("cnn_training_curves.png" if curve_choice == "Custom CNN" else "mobilenet_training_curves.png")

        if curve_path.exists():
            st.image(str(curve_path), caption=f"{curve_choice} Accuracy and Loss Curves", use_container_width=True)
        else:
            st.info("Training curve plot is generated upon training completion.")

    with tab_dataset:
        st.markdown("#### Dataset Balance & Proportion Analysis")
        dist_chart = config.GRAPHS_DIR / "category_distribution.png"
        if dist_chart.exists():
            st.image(str(dist_chart), caption="TrashNet Class Frequencies across 2,527 Images", use_container_width=True)

        dist_df = explore_dataset(config.DATA_DIR)
        st.dataframe(dist_df, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 4: ABOUT THE PROJECT
# ==========================================
elif selected_page == "ℹ️ About the Project":
    st.markdown("## ℹ️ Academic Project Documentation")
    st.markdown(
        "A technical breakdown of the algorithms, datasets, and design patterns utilized in this Smart Waste Classification project."
    )

    t1, t2, t3, t4 = st.tabs(["🧠 Deep Learning Architectures", "🚀 Transfer Learning & MobileNetV2", "📊 Dataset Details", "🛠️ Engineering Workflow"])

    with t1:
        st.markdown("### Convolutional Neural Networks (CNN)")
        st.markdown(
            r"""
            A **Convolutional Neural Network (CNN)** is a specialized class of deep neural networks designed to process grid-structured data like images.
            Traditional Multi-Layer Perceptrons (MLPs) suffer from parameter explosion when dealing with high-resolution image inputs and fail to preserve spatial hierarchy.

            **Core CNN Components in our Model:**
            1. **Convolutional Layers (`Conv2D`):** Slide learnable filter kernels across the input image to capture localized visual primitives (edges, corners, color gradients in early layers; complex textures and shapes in deeper layers).
            2. **Activation Function (`ReLU`):** Rectified Linear Unit introduces non-linearity, enabling the network to learn complex patterns without gradient degradation.
            3. **Batch Normalization (`BatchNorm`):** Normalizes activations across batches to stabilize gradient flow, accelerate convergence, and reduce internal covariate shift.
            4. **Max Pooling (`MaxPooling2D`):** Subsamples feature maps (typically 2×2 windows), reducing spatial dimensions, parameter count, and introducing spatial translation invariance.
            5. **Regularization (`Dropout` & L2 Weight Decay):** Randomly zeroes out neurons during training (20%-35%) to combat overfitting on small datasets.
            6. **Global Average Pooling (`GAP`):** Computes the average activation for each feature map, replacing massive flattened dense layers and drastically cutting parameter overhead.
            7. **Softmax Output Layer:** Normalizes output logits into an interpretable probability distribution satisfying $\sum_{i=1}^6 P(C_i) = 1.0$.
            """
        )

    with t2:
        st.markdown("### Transfer Learning & MobileNetV2 Architecture")
        st.markdown(
            """
            **What is Transfer Learning?**
            Training deep neural networks from scratch requires hundreds of thousands of labeled samples and extensive GPU hours.
            **Transfer Learning** circumvents this data dependency by leveraging weights pre-trained on massive benchmark datasets (e.g., ImageNet with 1.4 million images across 1,000 categories).
            The general feature extractors learned on ImageNet (edges, geometries, surfaces) transfer seamlessly to domain-specific waste classification.

            **MobileNetV2 Architectural Innovations:**
            - **Depthwise Separable Convolutions:** Splitting standard convolutions into depthwise (per-channel spatial filtering) and pointwise (1×1 cross-channel combination) reduces computational operations (FLOPs) by 8×-9× with minimal accuracy loss.
            - **Inverted Residual Blocks:** Rather than compressing then expanding channels, MobileNetV2 expands to high dimensions, applies depthwise convolution, then projects back to a lower-dimensional bottleneck.
            - **Linear Bottlenecks:** Uses linear activations in low-dimensional output layers to prevent non-linearities from destroying useful manifold representations.

            **Two-Stage Fine-Tuning Strategy Employed:**
            - **Phase 1 (Feature Extraction):** The pre-trained MobileNetV2 backbone is frozen (`trainable = False`). Only the custom classification head is trained with an Adam optimizer ($lr = 5 \times 10^{-4}$).
            - **Phase 2 (Fine-Tuning):** The top 35 layers of MobileNetV2 are unfrozen with a diminished learning rate ($lr = 5 \times 10^{-5}$) to adapt higher-level semantic filters to waste textures without catastrophic forgetting.
            """
        )

    with t3:
        st.markdown("### The TrashNet Dataset")
        st.markdown(
            """
            The **TrashNet** benchmark dataset was compiled by Gary Thung and Mindy Yang (Stanford University).
            It comprises photographic images of solid municipal waste captured against consistent white paper backgrounds under sunlight and artificial room lighting.

            **Dataset Characteristics:**
            - **Total Samples:** 2,527 images
            - **Category Breakdown:**
              - 📄 Paper: 594 images (23.5%)
              - 🍾 Glass: 501 images (19.8%)
              - 🧴 Plastic: 482 images (19.1%)
              - 🥫 Metal: 410 images (16.2%)
              - 📦 Cardboard: 403 images (16.0%)
              - 🗑️ Trash (residual non-recyclables): 137 images (5.4%)
            - **Class Imbalance Mitigation:**
              Because the 'Trash' category has only 137 samples vs 594 in 'Paper', we compute balanced class weights using Scikit-Learn:
              $$w_j = \\frac{N_{total}}{K \\times N_j}$$
              This penalizes classification mistakes on minority classes proportionately.
            - **Data Leakage Prevention:**
              Data is partitioned into **Train (70%)**, **Validation (15%)**, and **Test (15%)** splits using stratified sampling *before* any data augmentation is applied.
            """
        )

    with t4:
        st.markdown("### Project Architecture & Workflow")
        st.markdown(
            """
            ```
            ┌──────────────────┐
            │ TrashNet Dataset │ (2,527 Images)
            └─────────┬────────┘
                      │ Stratified 70:15:15 Split
                      ▼
            ┌───────────────────────────┬───────────────────────────┐
            │ Training Set (1,768)      │ Validation Set (379)      │ Test Set (380)
            │ [With Data Augmentation]  │ [Deterministic Resizing]  │ [Independent Benchmark]
            └─────────────┬─────────────┴─────────────┬─────────────┴───────────┬───────────┘
                          │                           │                         │
                          ▼                           ▼                         │
            ┌───────────────────────────┐ ┌───────────────────────────┐         │
            │ Model A: Custom Deep CNN  │ │ Model B: MobileNetV2 TL   │         │
            │ (Conv2D, BatchNorm, GAP)  │ │ (Frozen Base + Fine-Tune) │         │
            └─────────────┬─────────────┘ └─────────────┬─────────────┘         │
                          │                             │                       │
                          └──────────────┬──────────────┘                       │
                                         ▼                                      ▼
                          ┌───────────────────────────┐        ┌────────────────────────────┐
                          │ Saved Model Artifacts     │◄───────┤ Final Evaluation & Metrics │
                          │ (.keras files)            │        │ (CM, PR, F1, Loss Graphs)  │
                          └──────────────┬────────────┘        └────────────────────────────┘
                                         ▼
                          ┌───────────────────────────┐
                          │ Streamlit Web Dashboard   │
                          │ (Live Prediction & UI)    │
                          └───────────────────────────┘
            ```
            """
        )
