# ♻️ Smart Waste Classification Using Deep Learning (CNN & MobileNetV2)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

An end-to-end Computer Vision and Deep Learning system designed to automate municipal solid waste segregation across six distinct material categories: **Cardboard**, **Glass**, **Metal**, **Paper**, **Plastic**, and **Trash**.

The project implements, benchmarks, and contrasts a **Custom Deep Convolutional Neural Network (CNN)** built from scratch with an advanced **Transfer Learning architecture based on MobileNetV2** pre-trained on ImageNet. It includes an independent 380-sample test set evaluation suite, multi-image real-world testing, Grad-CAM explainability, and an interactive **Streamlit** dashboard.

---

## 📌 Problem Statement

Global municipal solid waste generation is projected to reach 3.4 billion metric tons annually by 2050. Conventional manual waste sorting is labor-intensive, hazardous to human sanitation workers, prone to fatigue and human error, and economically inefficient. Misclassified recyclables contaminate entire processing batches, causing recoverable material to be diverted into landfills and incinerators. Automated, vision-based waste classification utilizing Deep Learning provides an accurate, contactless, and scalable solution for smart recycling bins, automated sorting facilities, and IoT municipal waste infrastructure.

---

## 🎯 Objectives

1. **Automated Material Classification:** Accurately classify waste images into six primary categories (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
2. **Architecture Comparison:** Empirically compare a baseline custom 4-block deep CNN against a fine-tuned MobileNetV2 model under identical training conditions.
3. **Data Leakage Prevention:** Implement a strict stratified split (70% train, 15% validation, 15% test) with runtime data augmentation confined solely to the training set.
4. **Class Imbalance Mitigation:** Compute balanced cross-entropy class weights to prevent majority class bias.
5. **Visual Explainability (Grad-CAM):** Dynamically extract gradients from the final convolutional layers to render 2D activation heatmaps and explain where each model focuses its attention.
6. **Real-World Generalization Testing:** Provide a dedicated testing ground for uncurated external photographs, evaluating multi-model agreement/disagreement and distinguishing controlled benchmark scores from real-world testing.
7. **Deployable User Interface:** Package all capabilities into a responsive 6-page Streamlit web application.

---

## 🗂️ Six Waste Categories

| Category | Description | Common Real-World Examples | Disposal / Recycling Protocol |
| :--- | :--- | :--- | :--- |
| **Cardboard** | Corrugated boxes, packaging materials | Shipping cartons, cereal boxes, egg cartons | Flatten, keep completely dry, place in blue bin |
| **Glass** | Transparent and tinted silica containers | Beverage bottles, food jars, cosmetic glassware | Rinse clean, remove plastic/metal caps, recycle in glass depot |
| **Metal** | Ferrous and non-ferrous metals | Aluminum beverage cans, food tins, tin foil | Rinse thoroughly, crush aluminum cans to save volume |
| **Paper** | Cellulose fiber products | Office documents, newspapers, magazines, mail | Keep clean and dry; shred confidential documents |
| **Plastic** | Rigid and flexible polymer items | PET water bottles, HDPE milk jugs, shampoo bottles | Empty fluids, rinse, keep caps loosely secured |
| **Trash** | Non-recyclable composite or hazardous waste | Soiled napkins, plastic wrappers, ceramics | Discard in general landfill bin |

---

## 📊 Dataset Description & Source

The model is trained on the benchmark **TrashNet Dataset**:
- **Dataset Repository:** [https://github.com/garythung/trashnet](https://github.com/garythung/trashnet)
- **Total Images:** 2,527 real-world photographs taken under varied lighting and white background conditions.
- **Image Resolution:** Originally 512 × 384 pixels, resized to standard 224 × 224 for model ingestion.
- **Class Distribution:**
  - `paper`: 594 images (23.5%)
  - `glass`: 501 images (19.8%)
  - `plastic`: 482 images (19.1%)
  - `metal`: 410 images (16.2%)
  - `cardboard`: 403 images (15.9%)
  - `trash`: 137 images (5.4%)

### Controlled Benchmark vs. Real-World In-the-Wild Images
> [!IMPORTANT]
> **Controlled Benchmark:** TrashNet photographs were captured under standardized conditions (single, centered waste item placed on a plain white paper background under direct artificial/sunlight).  
> **Real-World / In-the-Wild Context:** External images from moving conveyor belts or consumer cellphones feature cluttered backgrounds, multiple objects, variable angles, and variable lighting. Softmax probabilities over a closed 6-class set do not constitute absolute proof that an arbitrary external image belongs to the waste domain; the system includes confidence thresholds and entropy warnings for human-in-the-loop validation.

---

## 🛠️ Technology Stack

- **Programming Language:** Python 3.10+ (tested on Python 3.12)
- **Deep Learning Framework:** TensorFlow 2.16+, Keras 3
- **Architectures:** Custom CNN (4 Conv Blocks), MobileNetV2 (ImageNet Transfer Learning)
- **Scientific Computing & Data Analysis:** NumPy, Pandas, Scikit-learn
- **Computer Vision & Image Processing:** OpenCV (`cv2`), Pillow (PIL)
- **Data Visualization & Charts:** Plotly Express, Matplotlib, Seaborn
- **Web Application & UI:** Streamlit

---

## 🏗️ Model Architectures

### 1. Custom Deep CNN (Model A)
Built from scratch to serve as a domain-specific baseline:
- **Input Layer:** 224 × 224 × 3 normalized image tensors.
- **Feature Extraction:** 4 sequential Convolutional Blocks:
  - Block 1: 32 filters (3×3), Batch Normalization, ReLU, Max Pooling (2×2), Dropout (0.20)
  - Block 2: 64 filters (3×3), Batch Normalization, ReLU, Max Pooling (2×2), Dropout (0.25)
  - Block 3: 128 filters (3×3), Batch Normalization, ReLU, Max Pooling (2×2), Dropout (0.30)
  - Block 4: 256 filters (3×3) (`conv4_1`), Batch Normalization, ReLU, Max Pooling (2×2), Dropout (0.35)
- **Dimensionality Reduction:** Global Average Pooling 2D (`gap`) instead of Flattening to eliminate parameter explosion.
- **Classification Head:** Dense layer (256 units, L2 regularization $10^{-4}$), Batch Normalization, ReLU, Dropout (0.30), Dense (128 units), Dropout (0.15), Softmax output layer (6 units).
- **Total Parameters:** 536,998 (535,526 trainable, 1,472 non-trainable).
- **Disk File Size:** 6.26 MB (`saved_models/cnn_model.keras`).

### 2. MobileNetV2 Transfer Learning (Model B)
Leverages deep visual representations pre-trained on ImageNet (1.4 million images):
- **Backbone:** MobileNetV2 featuring Depthwise Separable Convolutions and Inverted Residual Blocks with Linear Bottlenecks.
- **Two-Phase Training Pipeline:**
  - **Phase 1 (Feature Extraction):** Backbone weights are frozen (`trainable = False`). Only the custom top dense classification head is trained with learning rate $5 \times 10^{-4}$.
  - **Phase 2 (Fine-Tuning):** The top 35 layers of MobileNetV2 are unfrozen and trained end-to-end at a reduced learning rate ($5 \times 10^{-5}$) to adapt high-level visual features to waste surfaces.
- **Classification Head:** Global Average Pooling 2D (`gap`), Batch Normalization, Dense (256 units, ReLU, L2), Dropout (0.35), Dense (128 units), Dropout (0.21), Dense (6 units, Softmax).
- **Total Parameters:** 2,625,734 (1,972,742 trainable, 652,992 non-trainable).
- **Disk File Size:** 25.67 MB (`saved_models/mobilenet_model.keras`).

---

## 📈 Measured Model Comparison & Evaluation

Both models were independently benchmarked on the identical 380-sample test set (never seen during training or validation). All values below represent actual measurements:

| Evaluation Metric | Custom CNN (Model A) | MobileNetV2 (Model B) | Advantage |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | **58.68%** | **80.26%** | **+21.58% (MobileNetV2)** |
| **Macro Precision** | 55.93% | **82.99%** | **+27.06% (MobileNetV2)** |
| **Macro Recall** | 53.15% | **74.60%** | **+21.45% (MobileNetV2)** |
| **Macro F1-Score** | 52.49% | **76.78%** | **+24.29% (MobileNetV2)** |
| **Weighted F1-Score** | 57.03% | **80.14%** | **+23.11% (MobileNetV2)** |
| **Average Latency / Image** | **16.99 ms** | 39.39 ms | **+2.3× faster (Custom CNN)** |
| **Training Duration** | **295.4 s (~4.9 min)** | 549.5 s (~9.2 min) | Faster convergence |
| **Model File Size** | **6.26 MB** | 25.67 MB | Both < 30 MB (No Git LFS needed) |
| **Total Parameters** | **536,998** | 2,625,734 | Compact embedded footprint |
| **Trainable Parameters** | **535,526** | 1,972,742 | 75% of MobileNet params fine-tuned |

*Benchmark artifacts saved in:* `results/metrics/model_comparison.json` and `results/metrics/model_comparison.csv`.

---

## 🔬 Grad-CAM (Class Activation Mapping) Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual interpretability into deep neural network decisions:

1. **Gradient Extraction:** Computes the gradient of the predicted class score $y^c$ with respect to the feature activation maps $A^k$ of the final convolutional layer (`conv4_1` for Custom CNN; `out_relu` for MobileNetV2):
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. **Channel Weighting & ReLU:** Multiplies activations by weights $\alpha_k^c$ and filters positive contributors using ReLU:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
3. **Superimposed Visuals:** Resizes the 2D activation matrix to $224 \times 224$ and blends it over the original photograph using an adjustable opacity slider and OpenCV colormap.
4. **Target Hypothesis Selection:** Users can explain the top predicted class or select any alternative class (e.g. *Plastic* vs *Glass*) to inspect what spatial regions argue for or against that category.

---

## 🧪 Real-World Multi-Image Testing

The dedicated **Real-World Testing** page provides:
- **Multiple Image Uploads:** Batch upload multiple uncurated JPG/PNG photographs.
- **Parallel Inference:** Evaluates both Custom CNN and MobileNetV2 simultaneously on each uploaded item.
- **Model Consensus Detection:** Automatically reports whether models **Agree** (`🟢 Consensus`) or **Disagree** (`🟠 Disagreement`).
- **Complete 6-Class Probabilities:** Side-by-side probability breakdown for both architectures.
- **Session-Based History:** Tracks all tests conducted during the current browser session with a summary agreement ratio metric.
- **Data Protection:** External images are strictly utilized for inference and are **never** added to the training set or used for model fine-tuning.

---

## 📂 Project Structure

```
smart-waste-management-system/
│
├── app.py                          # Streamlit interactive web dashboard (6 pages)
├── requirements.txt                # Python project dependencies
├── README.md                       # Comprehensive documentation
├── config.py                       # Global paths, constants, and hyperparameters
├── sample_predict.py               # CLI sample inference verification script
├── run_all.py                      # Master pipeline runner
├── push_to_github.sh               # GitHub sync helper script
│
├── data/
│   ├── trashnet/                   # Full TrashNet dataset (downloaded on demand)
│   ├── split/                      # Stratified train (70%), val (15%), test (15%)
│   └── sample_test_images/         # 6 representative test samples for immediate demo
│
├── models/
│   ├── cnn_model.py                # Custom 4-block deep CNN architecture
│   └── mobilenet_model.py          # MobileNetV2 transfer learning & fine-tuning
│
├── training/
│   ├── preprocess.py               # Dataset acquisition, stratified split, generators
│   ├── train_cnn.py                # Custom CNN training & checkpointing script
│   ├── train_mobilenet.py          # MobileNetV2 two-phase training script
│   └── evaluate.py                 # Benchmarking with 10 actual measured metrics
│
├── utils/
│   ├── prediction.py               # Multi-model inference & entropy uncertainty
│   ├── gradcam.py                  # Dynamic Grad-CAM heatmap & overlay computation
│   └── visualization.py            # Matplotlib/Seaborn curve & matrix generators
│
├── saved_models/
│   ├── cnn_model.keras             # Trained Custom CNN weights (6.26 MB)
│   └── mobilenet_model.keras       # Trained MobileNetV2 weights (25.67 MB)
│
└── results/
    ├── graphs/                     # Training curves & comparison bar charts
    ├── confusion_matrices/         # Normalized & raw confusion matrix heatmaps
    └── metrics/                    # JSON reports & CSV comparison tables
```

---

## 🚀 Setup & Execution Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Ariha910/smart-waste-management-system.git
cd smart-waste-management-system

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Comprehensive Model Evaluation

```bash
python training/evaluate.py
```
*Calculates all 10 evaluation metrics, generates confusion matrices, and updates comparison tables.*

### 3. Launch the Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

**Application Navigation (6 Pages):**
- 🏠 **Home:** Project abstract, 4-step processing pipeline, and interactive category directive cards.
- 🔍 **Image Classification:** Single-image classification with 6-class probability distribution, top-class highlighting, and uncertainty warnings.
- 📊 **Model Performance:** Real-time accuracy metrics, comparative evaluation tables, normalized confusion matrices, and training trajectory curves.
- 🧪 **Real-World Testing:** Multi-image upload, simultaneous CNN vs MobileNetV2 evaluation, consensus detection, and session history.
- 🔬 **Grad-CAM Explainability:** 2D gradient heatmaps and superimposed overlays with model and target class selectors.
- ℹ️ **About Project:** Technical breakdown of CNNs, Transfer Learning, MobileNetV2, Grad-CAM math, and dataset details.

---

## 🔮 Future Improvements

1. **Edge Deployment:** Quantize the MobileNetV2 model to TensorFlow Lite (TFLite INT8) for microcontrollers (Raspberry Pi 5 / NVIDIA Jetson Orin Nano).
2. **Object Detection:** Migrate from single-image classification to YOLOv8 for real-time bounding box multi-item detection on moving conveyor belts.
3. **Dataset Expansion:** Incorporate e-waste, organic compostables, and hazardous medical waste classes.
4. **Sensor Fusion:** Combine visual classification with inductive proximity sensors and load cells for multi-modal waste verification.

---

## 📜 License & Acknowledgments
- **Dataset:** [TrashNet by Gary Thung & Mindy Yang](https://github.com/garythung/trashnet).
- Built for academic research and environmental sustainability.
