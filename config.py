"""
Configuration module for Smart Waste Classification.
Defines paths, class definitions, hyperparameters, and styling constants.
"""

import os
from pathlib import Path

# Set cache directories to writable locations
os.environ["KERAS_HOME"] = "/tmp/keras"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data" / "trashnet"
DATA_SPLIT_DIR = BASE_DIR / "data" / "split"
TRAIN_DIR = DATA_SPLIT_DIR / "train"
VAL_DIR = DATA_SPLIT_DIR / "val"
TEST_DIR = DATA_SPLIT_DIR / "test"

# Output directories
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
CNN_MODEL_PATH = SAVED_MODELS_DIR / "cnn_model.keras"
MOBILENET_MODEL_PATH = SAVED_MODELS_DIR / "mobilenet_model.keras"

RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
CM_DIR = RESULTS_DIR / "confusion_matrices"
METRICS_DIR = RESULTS_DIR / "metrics"
COMPARISON_TABLE_PATH = METRICS_DIR / "model_comparison.json"

# Ensure runtime directories exist
for directory in [
    DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    SAVED_MODELS_DIR,
    GRAPHS_DIR,
    CM_DIR,
    METRICS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset specifications
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASSES)

# Model hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

BATCH_SIZE = 32
RANDOM_STATE = 42

# Training ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Hyperparameters
CNN_EPOCHS = 20
CNN_LR = 0.001

MOBILENET_BASE_EPOCHS = 8
MOBILENET_FINE_EPOCHS = 6
MOBILENET_BASE_LR = 0.0005
MOBILENET_FINE_LR = 0.00005

# Category metadata for UI and Education
CATEGORY_INFO = {
    "cardboard": {
        "title": "Cardboard",
        "icon": "📦",
        "badge": "Recyclable",
        "color": "#8d6e63",
        "bg_color": "#efebe9",
        "description": "Corrugated boxes, shipping cartons, and paperboard packaging.",
        "recycling_tip": "Flatten boxes before placing in the blue or yellow recycling bin. Keep dry and remove any plastic tapes or bubble wrap.",
        "environmental_impact": "Recycling 1 ton of cardboard saves 9 cubic yards of landfill space and 46 gallons of oil.",
    },
    "glass": {
        "title": "Glass",
        "icon": "🍾",
        "badge": "100% Recyclable",
        "color": "#0288d1",
        "bg_color": "#e1f5fe",
        "description": "Bottles, jars, beverage containers, and culinary glass containers.",
        "recycling_tip": "Rinse thoroughly to remove food or liquid residue. Caps and corks should be removed. Avoid broken window glass or mirrors.",
        "environmental_impact": "Glass is 100% recyclable and can be recycled endlessly without loss in quality or purity.",
    },
    "metal": {
        "title": "Metal",
        "icon": "🥫",
        "badge": "High Value Recyclable",
        "color": "#78909c",
        "bg_color": "#eceff1",
        "description": "Aluminum cans, tin cans, foil trays, and metal bottle caps.",
        "recycling_tip": "Empty and lightly rinse cans. Labels can usually stay on. Do not crush cans if your local MRF uses optical sorters.",
        "environmental_impact": "Recycling aluminum uses 95% less energy than producing primary aluminum from bauxite ore.",
    },
    "paper": {
        "title": "Paper",
        "icon": "📄",
        "badge": "Recyclable",
        "color": "#43a047",
        "bg_color": "#e8f5e9",
        "description": "Newspapers, office paper, notebooks, envelopes, and brochures.",
        "recycling_tip": "Must be clean and dry. Paper soiled with grease or oils (e.g., pizza boxes) should go to compost or trash.",
        "environmental_impact": "Recycling 1 ton of paper preserves 17 mature trees, 7,000 gallons of water, and 3 cubic yards of landfill.",
    },
    "plastic": {
        "title": "Plastic",
        "icon": "🧴",
        "badge": "Conditionally Recyclable",
        "color": "#fb8c00",
        "bg_color": "#fff3e0",
        "description": "PET bottles, HDPE detergent jugs, containers, cups, and food tubs.",
        "recycling_tip": "Check resin identification code (PET 1 and HDPE 2 are universally accepted). Rinse clean and replace caps.",
        "environmental_impact": "Conventional plastics take 450+ years to decompose in landfills and cause massive marine pollution.",
    },
    "trash": {
        "title": "Trash / Residual Waste",
        "icon": "🗑️",
        "badge": "Non-Recyclable",
        "color": "#e53935",
        "bg_color": "#ffebee",
        "description": "Non-recyclable composites, broken ceramics, dirty wrappers, sanitary waste.",
        "recycling_tip": "Dispose of in the general municipal landfill or waste-to-energy incinerator bin. Minimize use of non-recyclables.",
        "environmental_impact": "Landfill waste produces methane, a greenhouse gas 28 times more potent than carbon dioxide.",
    },
}
