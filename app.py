import pandas as pd
import plotly.express as px
import streamlit as st

import config
from utils.prediction import predict_waste

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #f5f7f5; }
.block-container { max-width: 1180px; padding: 2rem 1.25rem 3rem; }
.topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.25rem; }
.brand { display:flex; align-items:center; gap:.65rem; }
.brandmark { width:42px;height:42px;border-radius:12px;background:#064e3b;color:#fff;display:grid;place-items:center;font-size:22px; }
.brandname { font-size:1.18rem;font-weight:800;color:#15352b; }
.status { color:#52705f;font-size:.78rem;font-weight:700;background:#e9f4ed;border:1px solid #d3e7da;border-radius:999px;padding:.38rem .7rem; }
.hero { background:#064e3b;border-radius:18px;padding:2rem 2.2rem;color:white;margin-bottom:1.25rem; }
.hero h1 { font-size:2.15rem;line-height:1.12;margin:.15rem 0 .55rem;letter-spacing:-.03em; }
.hero p { max-width:680px;color:#d9eee3;line-height:1.55;margin:0;font-size:.96rem; }
.panel { background:white;border:1px solid #e2e9e4;border-radius:16px;padding:1.25rem;box-shadow:0 4px 18px rgba(16,53,38,.055); }
.panel-title { color:#15352b;font-size:.88rem;font-weight:800;margin:0 0 .75rem; }
[data-testid="stFileUploader"] section { border:1.5px dashed #90b39f!important;border-radius:14px!important;background:#f8fbf9!important;min-height:230px; }
[data-testid="stFileUploader"] section:hover { border-color:#087b50!important; }
.stButton>button { border-radius:10px!important;font-weight:750!important;height:44px; }
.stButton>button[kind="primary"] { background:#087b50!important;border-color:#087b50!important;color:white!important; }
.result { border:1px solid #cfe0d5;border-radius:16px;padding:1.35rem;background:#f4faf6; }
.result-label { color:#52705f;font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em; }
.result-name { color:#064e3b;font-size:2rem;font-weight:800;margin:.28rem 0 .2rem; }
.result-confidence { color:#52675d;font-size:.92rem; }
.empty { border:1px dashed #d7e1da;border-radius:16px;min-height:230px;display:grid;place-items:center;text-align:center;color:#789083;background:#fafcfb;padding:1.5rem; }
@media (max-width:760px) {
 .block-container { padding:1rem .75rem 2rem; }
 .hero { padding:1.35rem; }
 .hero h1 { font-size:1.65rem; }
 .status { display:none; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div class="brand"><div class="brandmark">♻</div><div class="brandname">EcoSort AI</div></div>
  <div class="status">● Model engine ready</div>
</div>
<div class="hero">
  <h1>Smart Waste Classification</h1>
  <p>Upload a waste image to classify it using a trained CNN or MobileNetV2 model.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.08, .92], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">1  Upload an image</p>', unsafe_allow_html=True)

    model_key = st.selectbox(
        "Choose model",
        options=["mobilenet", "cnn"],
        format_func=lambda x: "MobileNetV2" if x == "mobilenet" else "Custom CNN",
    )

    uploaded = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    preview = None
    if uploaded:
        from PIL import Image
        preview = Image.open(uploaded).convert("RGB")
        st.image(preview, use_container_width=True, caption=uploaded.name)
    else:
        st.markdown('<div class="empty"><div><div style="font-size:2rem">↥</div><b>Drop an image here</b><br><span style="font-size:.82rem">or use Browse files</span></div></div>', unsafe_allow_html=True)

    predict_clicked = st.button("Predict Image", type="primary", use_container_width=True, disabled=uploaded is None)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">2  Classification result</p>', unsafe_allow_html=True)

    if predict_clicked and preview is not None:
        model_path = config.MOBILENET_MODEL_PATH if model_key == "mobilenet" else config.CNN_MODEL_PATH
        if not model_path.exists():
            st.error("The selected model file is not available. Please train or restore the model first.")
        else:
            with st.spinner("Analyzing image..."):
                try:
                    result = predict_waste(preview, model_type=model_key, confidence_threshold=.45)

                    predicted = result["predicted_class"]
                    info = result["category_info"]
                    confidence = result["confidence"]

                    st.markdown(f"""
                    <div class="result">
                      <div class="result-label">Predicted category</div>
                      <div class="result-name">{info.get('icon','♻')} {info.get('title', predicted.title())}</div>
                      <div class="result-confidence">Confidence <b>{confidence:.1%}</b>  ·  {result['model_used']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if result["is_uncertain"]:
                        st.warning(result["warning_message"])

                    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
                    prob_rows = [
                        {"Category": cls.title(), "Probability": result["probabilities"][cls]}
                        for cls in config.CLASSES
                    ]
                    prob_df = pd.DataFrame(prob_rows).sort_values("Probability", ascending=True)

                    fig = px.bar(
                        prob_df,
                        x="Probability",
                        y="Category",
                        orientation="h",
                        text=prob_df["Probability"].map(lambda v: f"{v:.1%}"),
                        color="Probability",
                        color_continuous_scale=["#dce9e1", "#087b50"],
                        range_x=[0, 1],
                    )
                    fig.update_layout(
                        height=280,
                        margin=dict(l=0, r=10, t=8, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False,
                        xaxis=dict(tickformat=".0%", gridcolor="#edf2ef"),
                        yaxis=dict(title=None),
                        font=dict(family="Plus Jakarta Sans", size=12, color="#52675d"),
                    )
                    fig.update_traces(textposition="outside", cliponaxis=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(info.get("recycling_tip", "Follow local recycling guidance."))

                except Exception as exc:
                    st.error(f"Could not classify this image: {exc}")
    else:
        st.markdown('<div class="empty"><div><div style="font-size:2rem">◉</div><b>Waiting for an image</b><br><span style="font-size:.82rem">Your prediction and confidence will appear here</span></div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
