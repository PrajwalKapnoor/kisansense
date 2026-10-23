# KisanSense — Complete Project File Directory & Architecture

This document provides a comprehensive map of the entire **KisanSense** repository, detailing the purpose, dependencies, and roles of each directory and file.

---

## 1. Directory Tree Overview

```
kisansense/
├── .gitignore                                 # Git ignore rules for venv, cache, datasets, artifacts
├── README.md                                  # Project overview, module descriptions, quickstart
├── requirements.txt                           # Project dependencies
├── PROJECT_STRUCTURE.md                       # Comprehensive directory map and architecture guide
│
├── app/                                       # Streamlit Multi-page Frontend
│   ├── __init__.py
│   ├── Home.py                                # Landing page with module navigation & health status
│   ├── api_client.py                          # HTTP client communicating with the FastAPI backend
│   ├── components.py                          # Reusable UI widgets, layout cards & Grad-CAM visualizers
│   └── pages/
│       ├── 1_Disease_Check.py                 # Leaf disease diagnostics with Grad-CAM heatmap
│       ├── 2_Crop_Advisor.py                  # Soil & weather crop recommendation interface
│       └── 3_Price_Explorer.py                # Mandi price time series forecast & backtest explorer
│
├── backend/                                   # FastAPI RESTful Microservice
│   ├── __init__.py
│   └── app/
│       ├── __init__.py
│       ├── main.py                            # FastAPI entry point, CORS, routers & /health check
│       ├── deps.py                            # Dependency injection & lazy model loading (503 handling)
│       ├── schemas.py                         # Pydantic request/response validation schemas
│       └── routers/
│           ├── crop.py                        # /api/v1/crop endpoints
│           ├── disease.py                     # /api/v1/disease endpoints (file upload & Grad-CAM)
│           ├── metrics.py                     # /api/v1/metrics endpoints (evaluation benchmarks)
│           └── price.py                       # /api/v1/price endpoints (7-day forecast & history)
│
├── src/                                       # Core Reusable Machine Learning Library
│   └── kisansense/
│       ├── __init__.py
│       ├── common/                            # Shared Utilities & Configuration
│       │   ├── __init__.py
│       │   ├── config.py                      # Central file paths, seeds, feature bounds
│       │   └── seed.py                        # Reproducibility seed setter (numpy, random, tf)
│       ├── crop/                              # Crop Recommendation Module (Tabular ML)
│       │   ├── __init__.py
│       │   ├── train.py                       # LogisticRegression vs RandomForest training & evaluation
│       │   └── predict.py                     # Top-3 predictions with probabilities & SHAP explainability
│       ├── disease/                           # Tomato Leaf Disease Module (Deep Learning)
│       │   ├── __init__.py
│       │   ├── data.py                        # Image loaders, preprocessing & augmentations
│       │   ├── model.py                       # CNN baseline and MobileNetV2 transfer learning
│       │   ├── gradcam.py                     # Grad-CAM visual explainability heatmap generator
│       │   ├── train.py                       # Deep learning training loop with early stopping
│       │   ├── predict.py                     # Inference pipeline with confidence & uncertainty thresholds
│       │   └── test_disease.py                # Direct unit tests for disease routines
│       └── price/                             # Mandi Price Forecasting Module (Time Series)
│           ├── __init__.py
│           ├── data.py                        # Karnataka mandi dataset loader & temporal indexing
│           ├── features.py                    # Lag features, rolling statistics, calendar indicators
│           ├── models.py                      # Naive baseline, ARIMA, & Gradient Boosting models
│           ├── backtest.py                    # Expanding / rolling-origin backtesting framework
│           ├── train.py                       # Model selection, hyperparameter tuning & evaluation
│           └── predict.py                     # 7-day price forecasting & trend classification
│
├── scripts/                                   # Automation & Operational Scripts
│   ├── check_datasets.py                      # Pre-flight data integrity & schema validation
│   ├── train_crop.py                          # CLI runner for training & saving crop recommendation models
│   └── train_price.py                         # CLI runner for backtesting & training price forecasting models
│
├── data/                                      # Datasets (Raw, Interim, Processed)
│   ├── README.md                              # Data documentation, origins, schemas & licensing
│   ├── raw/
│   │   ├── crop/
│   │   │   └── Crop_recommendation.csv        # Soil nutrients (N, P, K), weather (temp, humidity, rainfall)
│   │   ├── disease/                           # Target directory for PlantVillage tomato leaf images
│   │   └── price/
│   │       └── mandi_prices.csv               # Historical Karnataka APMC mandi tomato market prices
│   ├── interim/                               # Cached/partially processed intermediate data
│   └── processed/                             # Sanitized datasets ready for model consumption
│
├── artifacts/                                 # Trained Model Binaries & State
│   ├── crop_model.joblib                      # Serialized scikit-learn pipeline for crop advisor
│   └── price_model.joblib                     # Serialized time-series forecasting model
│
├── docs/                                      # Documentation & Validation Evidence
│   └── results/
│       ├── crop_confusion_matrix.png          # Visual confusion matrix for crop recommendation
│       ├── crop_metrics.json                  # Accuracy, precision, recall, and F1 scores
│       ├── price_backtest_rows.csv            # Detailed out-of-sample backtest prediction records
│       ├── price_mae_by_horizon.png           # Horizon-wise (1 to 7 days) MAE error curves
│       └── price_metrics.json                 # Test MAE, RMSE, MAPE comparison vs naive baseline
│
├── notebooks/                                 # Research, Exploration & Remote GPU Notebooks
│   └── disease_training_colab.ipynb           # Google Colab notebook for GPU training & export
│
└── tests/                                     # Automated Test Suite (Pytest)
    ├── __init__.py
    ├── test_api.py                            # FastAPI endpoint contract & integration tests
    ├── test_app.py                            # Streamlit AppTest UI logic & error state tests
    ├── test_crop.py                           # Crop recommendation pipeline & prediction tests
    └── test_price.py                          # Time-series backtesting & price prediction tests
```

---

## 2. Component Details

### 2.1 Web Applications & API Layer

| File / Folder | Role & Description |
| :--- | :--- |
| **`app/Home.py`** | Streamlit multi-page landing view with descriptive cards linking to each tool and live API backend readiness statuses. |
| **`app/pages/`** | Sub-pages loaded automatically by Streamlit: `1_Disease_Check.py` (image upload & Grad-CAM viewer), `2_Crop_Advisor.py` (soil parameter form & crop probabilities), and `3_Price_Explorer.py` (mandi price history & Vega-Lite forecast visualization). |
| **`app/api_client.py`** | Centralized client managing HTTP calls to FastAPI with standardized timeouts and user-friendly error wrapping. |
| **`app/components.py`** | Reusable UI components: headers, cards, disclaimer notices, and base64 Grad-CAM overlay renderers. |
| **`backend/app/main.py`** | FastAPI application initialization; mounts API routers with prefixes `/api/v1/crop`, `/api/v1/disease`, `/api/v1/price`, and `/api/v1/metrics`. |
| **`backend/app/deps.py`** | Modular dependency injection providing model instances; returns HTTP 503 Service Unavailable if model files have not yet been trained or placed in `artifacts/`. |
| **`backend/app/schemas.py`** | Strict Pydantic models for request validation (enforcing physiological ranges on N, P, K, pH, temperature, etc.) and response models. |

---

### 2.2 Core Library (`src/kisansense`)

The foundational code is structured into clean domain modules under `src/kisansense/`:

1. **`common/`**:
   - `config.py`: Root paths, artifact directory definitions, image size targets ($224 \times 224$), and input boundary sanity checks.
   - `seed.py`: Global seed enforcement across `random`, `numpy`, and `tensorflow`.
2. **`crop/`**:
   - `train.py`: Trains Logistic Regression vs. Random Forest models with stratified k-fold splits, saving the winning pipeline to `artifacts/crop_model.joblib`.
   - `predict.py`: Loads the trained pipeline and returns ranked crop recommendations with probabilistic confidence and feature importance explanations.
3. **`disease/`**:
   - `model.py` & `data.py`: Prepares image datasets and initializes MobileNetV2 feature extractors with custom dense classification heads.
   - `gradcam.py`: Computes gradient-weighted class activation maps (Grad-CAM) to generate explainable visual heatmaps highlighting leaf lesions.
   - `predict.py`: Runs forward passes, applying uncertainty detection (flags predictions with low softmax confidence or high entropy).
4. **`price/`**:
   - `data.py` & `features.py`: Cleans mandi transactions, handles missing dates with forward fills, and generates autoregressive lag/rolling features.
   - `backtest.py`: Evaluates forecasting performance across historical expanding time horizons to prevent data leakage.
   - `predict.py`: Produces 7-day forward predictions, prediction confidence bounds, and trend directional alerts.

---

### 2.3 Automation & Utilities (`scripts/`)

- **`scripts/check_datasets.py`**: Validates the availability and structural integrity of all raw dataset files prior to executing model training.
- **`scripts/train_crop.py`**: CLI pipeline trigger that fits the crop recommendation model, computes cross-validation metrics, exports confusion matrices, and writes the model binary.
- **`scripts/train_price.py`**: CLI pipeline trigger executing time-series backtesting on Karnataka mandi prices and saving the best-performing model.

---

### 2.4 Testing Suite (`tests/`)

- **`tests/test_api.py`**: Tests FastAPI routes using FastAPI's `TestClient`, checking response schemas, status codes, and 503 degradation when models are missing.
- **`tests/test_app.py`**: Streamlit `AppTest` test cases that verify page rendering, user input validation, and alert components without requiring a live backend.
- **`tests/test_crop.py` & `tests/test_price.py`**: Unit tests verifying model training routines, feature engineering pipelines, boundary validation, and inference accuracy.
