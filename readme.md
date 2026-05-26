# 📉 End-to-End ML Project: Telco Customer Churn Prediction

> From raw data to production deployment with MLflow, FastAPI, Docker & AWS

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![MLflow](https://img.shields.io/badge/MLflow-tracking-orange)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)

---

## 🎯 Problem Statement

Customer churn is a critical business challenge — retaining a customer costs **5–25x less** than acquiring a new one.

This project builds a complete ML pipeline that predicts whether a telecom customer will churn, based on their demographics, contract type, usage patterns, and monthly charges.

- **Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 customers, 20+ features
- **Target variable:** `Churn` (Yes / No)

---

## 🏗️ Pipeline Architecture

```
Raw Data → Data Validation → Preprocessing → Feature Engineering → Model Training
                                                                         ↓
                                                   Model Packaging → Deployment → CI/CD → Monitoring
```

---

## 📁 Project Structure

```
├── .github/workflows/       # CI/CD automation
├── app/                     # FastAPI application
├── artifacts/               # Saved models and outputs
├── configs/                 # Configuration files
├── data/                    # Raw and processed data
├── docker/                  # Dockerfile and compose files
├── great_expectations/      # Data validation suite
├── mlruns/                  # MLflow experiment tracking
├── notebooks/
│   └── EDA.ipynb            # Exploratory Data Analysis
├── scripts/                 # Utility scripts
├── src/
│   ├── data/
│   │   ├── load_data.py     # Data ingestion
│   │   └── preprocess.py    # Preprocessing pipeline
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   ├── models/              # Training & evaluation (in progress)
│   └── utils/
│       └── validate_data.py # Great Expectations validation
└── tests/                   # Unit tests

```

---

## 🚧 Status: In Progress

This project is currently under active development. The data pipeline (loading, validation, preprocessing, and feature engineering) is complete. Model training, API deployment, and cloud infrastructure are coming next.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Validation | Great Expectations |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Containerization | Docker |
| Cloud Deployment | AWS (EC2 / ECS) |
| CI/CD | GitHub Actions |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the EDA notebook
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 📊 Dataset

The dataset is sourced from Kaggle and contains customer-level data including:
- **Demographics:** gender, senior citizen status, dependents
- **Services:** phone, internet, streaming, security
- **Contract:** type, billing method, paperless billing
- **Financials:** monthly charges, total charges
- **Target:** whether the customer churned


