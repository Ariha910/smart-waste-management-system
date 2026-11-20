# ♻️ Smart Waste Classification Using Deep Learning

An end-to-end Computer Vision and Deep Learning system designed to automate municipal solid waste segregation across six distinct material categories: **Cardboard**, **Glass**, **Metal**, **Paper**, **Plastic**, and **Trash**.

The project implements, benchmarks, and compares a **Custom Deep Convolutional Neural Network (CNN)** built from scratch with an advanced **Transfer Learning architecture based on MobileNetV2** pre-trained on ImageNet. It includes an independent evaluation suite, prediction confidence metrics, Grad-CAM explainability concepts, and an interactive **Streamlit** dashboard.

---

## 📌 Problem Statement

Global municipal solid waste generation is projected to reach 3.4 billion metric tons annually by 2050. Conventional manual waste sorting is labor-intensive, hazardous to human sanitation workers, prone to human error, and economically inefficient. Misclassified recyclables contaminate entire batches, diverting recyclable material into landfills and incinerators. Automated, vision-based waste classification utilizing Deep Learning offers an accurate, contactless, and scalable solution for smart recycling bins, automated sorting facilities, and IoT municipal waste infrastructure.

---

## 🎯 Objectives

1. **Automated Material Classification:** Accurately classify waste images into six primary categories (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
2. **Architecture Comparison:** Empirically compare a baseline custom 4-block deep CNN against a fine-tuned MobileNetV2 model under identical training conditions.
3. **Data Leakage Prevention:** Implement a strict stratified split (70% train, 15% validation, 15% test) with runtime data augmentation confined solely to the training set.
4. **Class Imbalance Mitigation:** Compute balanced cross-entropy class weights to prevent majority class bias.
5. **Model Explainability & Reliability:** Integrate prediction confidence scoring, Shannon entropy-based uncertainty alerts for out-of-distribution inputs, and Grad-CAM attention concepts.
6. **Deployable User Interface:** Package the models into an intuitive, responsive Streamlit web application.

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

The model is trained on the standard benchmark **TrashNet Dataset**:
- **Source:** [https://github.com/garythung/trashnet](https://github.com/garythung/trashnet)
- **Total Images:** 2,527 real-world photographs taken under varied lighting and white background conditions.
- **Image Resolution:** Originally 512 × 384 pixels, resized to standard 224 × 224 for model ingestion.
- **Class Distribution:**
  - `paper`: 594 images (23.5%)
  - `glass`: 501 images (19.8%)
  - `plastic`: 482 images (19.1%)
  - `cardboard`: 403 images (15.9%)
  - `metal`: 410 images (16.2%)
  - `trash`: 137 images (5.4%)

---

## 🛠️ Technology Stack

- **Programming Language:** Python 3.10+ (tested on Python 3.12)
- **Deep Learning Framework:** TensorFlow 2.16+, Keras 3
- **Architectures:** Custom CNN, MobileNetV2 (ImageNet Transfer Learning)
- **Scientific Computing & Data Analysis:** NumPy, Pandas, Scikit-learn
- **Computer Vision & Image Processing:** OpenCV, Pillow (PIL)
- **Data Visualization:** Matplotlib, Seaborn, Plotly Express
- **Web Application:** Streamlit

---

## 🏗️ Model Architectures

### 1. Custom Deep CNN (Model A)
Built from scratch to serve as a domain-specific baseline:
- **Input Layer:** 224 × 224 × 3 normalized image tensors.
- **Feature Extraction:** 4 sequential Convolutional Blocks:
  - Block 1: 32 filters (3×3), Batch Normalization, ReLU, Max Pooling (2×2), Dropout (0.25)
  - Block 2: 64 filters (3×3), Batch Normalization, ReLU, Max Pooling (2×2), Dropout (0.25)
  - Block 3: 128 filters (3×3), Batch Normalization, ReLU, Max Pooling (2×2), Dropout (0.30)
  - Block 4: 256 filters (3×3), Batch Normalization, ReLU, Max Pooling (2×2), Dropout (0.40)
- **Dimensionality Reduction:** Global Average Pooling 2D (GAP) instead of Flattening to prevent parameter explosion.
- **Classification Head:** Dense layer (128 units, L2 regularization $10^{-4}$), Batch Normalization, ReLU, Dropout (0.5), Softmax output layer (6 units).
- **Total Parameters:** 536,998 (2.05 MB unquantized).

### 2. MobileNetV2 Transfer Learning (Model B)
Leverages deep visual representations pre-trained on the 1.4-million-image ImageNet dataset:
- **Backbone:** MobileNetV2 featuring Depthwise Separable Convolutions and Inverted Residual Blocks with Linear Bottlenecks.
- **Two-Phase Training:**
  - **Phase 1 (Feature Extraction):** Backbone weights are frozen. Only the custom top dense classification head is trained with learning rate $10^{-3}$.
  - **Phase 2 (Fine-Tuning):** The top 35 layers of MobileNetV2 are unfrozen and trained end-to-end at a gentle learning rate ($5 \times 10^{-5}$) to adapt high-level visual features to waste surfaces.
- **Classification Head:** Global Average Pooling 2D, Batch Normalization, Dense (256 units, ReLU), Dropout (0.4), Dense (6 units, Softmax).
- **Total Parameters:** 2,625,734 (10.02 MB unquantized).

---

## 🔄 Data Preprocessing & Leakage Prevention

1. **Stratified Split:** 70% Training (1,768 images), 15% Validation (379 images), and 15% Testing (380 images) using stratified sampling to preserve original class ratios.
2. **Leakage Isolation:** Preprocessing statistics and data augmentations are strictly computed on the training set. Validation and test sets are never augmented.
3. **Data Augmentation:** Real-time geometric transformations applied to training data:
   - Random rotation ($\pm 20^\circ$)
   - Width and height shifts ($\pm 15\%$)
   - Shear transformations ($0.15$)
   - Random zoom ($[0.85, 1.15]$)
   - Horizontal flipping
   - Fill mode: nearest
4. **Normalization:** Pixel scaling from $[0, 255]$ to $[0, 1]$ (CNN) and $[-1, 1]$ (MobileNetV2).

---

## 📈 Evaluation & Benchmark Results

Both models were independently evaluated against the same 380-sample test set (never seen during training or validation).

### Model Comparison Table (Actual Test Results)

| Performance Metric | Custom CNN (Model A) | MobileNetV2 (Model B) | Advantage |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | **58.68%** | **80.26%** | **+21.58% (MobileNetV2)** |
| **Macro Precision** | 55.93% | **82.99%** | **+27.06% (MobileNetV2)** |
| **Macro Recall** | 53.15% | **74.60%** | **+21.45% (MobileNetV2)** |
| **Macro F1-Score** | 52.49% | **76.78%** | **+24.29% (MobileNetV2)** |
| **Weighted F1-Score** | 57.03% | **80.14%** | **+23.11% (MobileNetV2)** |
| **Total Parameters** | 536,998 | 2,625,734 | Compact footprint |
| **Model Size (.keras)** | 6.5 MB | 26.9 MB | Both < 30 MB (Standard Git) |

### Key Findings & Confusion Matrix Insights
1. **Transfer Learning Superiority:** MobileNetV2 achieves an **80.26% test accuracy**, outperforming the custom CNN by over 21 percentage points due to robust pre-learned feature filters.
2. **Class Confusion Patterns:** The primary classification bottleneck occurs between transparent plastic and clear glass bottles due to identical refractive and specular highlights. Cardboard vs Paper also presents subtle texture boundaries.
3. **Minority Class Handling:** Balanced loss weighting allowed the under-represented `trash` class (137 samples) to achieve viable recall rather than collapsing to majority-class prediction.

---

## 🔍 Prediction Confidence & Out-of-Distribution Handling

Standard Softmax activations can produce high confidence even on random, non-waste imagery. To prevent false classifications:
- **Confidence Threshold:** Any prediction with top probability $< 45\%$ triggers an explicit `Low Confidence` warning.
- **Shannon Entropy Analysis:** Computes the informational entropy across the probability distribution:
  $$H(P) = -\sum_{i=1}^{K} P_i \log_2(P_i)$$
- When entropy is flat or confidence is ambiguous, the system prompts for manual inspection rather than blindly automating disposal.

---

## 🔬 Grad-CAM Explainability (Class Activation Mapping)

Gradient-weighted Class Activation Mapping (Grad-CAM) explains deep neural network decisions by visualizing gradients flowing into the final convolutional layer:
1. Computes the gradient of the predicted class score $y^c$ with respect to feature activation maps $A^k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. Calculates a weighted combination of forward activation maps followed by ReLU:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
3. Produces a 2D heat-map overlaid onto the waste item, verifying that the model focuses on edges, labels, and reflective specularities rather than background artifacts.

---

## 📂 Project Structure

```
smart-waste-classification/
│
├── app.py                          # Streamlit interactive web dashboard
├── requirements.txt                # Python project dependencies
├── README.md                       # Comprehensive documentation
├── config.py                       # Global paths, constants, and hyperparameters
├── sample_predict.py               # CLI sample inference verification script
├── run_all.py                      # Master execution script
├── push_to_github.sh               # GitHub sync helper script
│
├── data/
│   ├── trashnet/                   # Raw TrashNet dataset (downloaded on demand)
│   ├── split/                      # Stratified train (70%), val (15%), test (15%)
│   └── sample_test_images/         # Representative samples for quick UI testing
│
├── models/
│   ├── cnn_model.py                # Custom 4-block deep CNN architecture
│   └── mobilenet_model.py          # MobileNetV2 transfer learning & fine-tuning
│
├── training/
│   ├── preprocess.py               # Dataset acquisition, stratified split, generators
│   ├── train_cnn.py                # Custom CNN training & checkpointing script
│   ├── train_mobilenet.py          # MobileNetV2 two-phase training script
│   └── evaluate.py                 # Independent benchmark & comparison generator
│
├── utils/
│   ├── prediction.py               # Inference pipeline & uncertainty estimation
│   └── visualization.py            # Matplotlib/Seaborn curve & matrix generators
│
├── saved_models/
│   ├── cnn_model.keras             # Trained Custom CNN weights (6.5 MB)
│   └── mobilenet_model.keras       # Trained MobileNetV2 weights (26.9 MB)
│
└── results/
    ├── graphs/                     # Training curves & comparison bar charts
    ├── confusion_matrices/         # Normalized & raw confusion matrix heatmaps
    └── metrics/                    # JSON reports & CSV comparison tables
```

---

## 🚀 Setup & Installation Instructions

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

### 2. Preprocess Dataset & Train Models (Optional if using pre-trained weights)

```bash
# Prepare dataset and stratified splits
python training/preprocess.py

# Train Custom CNN
python training/train_cnn.py

# Train MobileNetV2
python training/train_mobilenet.py

# Evaluate both models on test set
python training/evaluate.py
```

### 3. Launch the Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔮 Future Improvements

1. **Edge Deployment:** Quantize the MobileNetV2 model to TensorFlow Lite (TFLite) for microcontrollers (Raspberry Pi 5 / NVIDIA Jetson Nano).
2. **Object Detection:** Migrate from single-image classification to YOLOv8 for real-time bounding box multi-item detection on moving conveyor belts.
3. **Dataset Expansion:** Incorporate e-waste, organic compostables, and medical waste classes.
4. **Sensor Fusion:** Combine visual classification with inductive proximity sensors and load cells for multi-modal waste verification.

---

## 📜 License & Acknowledgments
- **Dataset:** [TrashNet by Gary Thung & Mindy Yang](https://github.com/garythung/trashnet).
- Built for academic research and environmental sustainability.
