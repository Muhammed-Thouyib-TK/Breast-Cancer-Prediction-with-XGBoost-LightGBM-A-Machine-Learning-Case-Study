# 🏥 Breast Cancer Prediction with XGBoost & LightGBM

This project demonstrates how powerful boosting algorithms like **XGBoost** and **LightGBM** can be applied to accurately predict breast cancer diagnoses based on clinical diagnostic features. From preprocessing to evaluation, this end-to-end machine learning pipeline showcases the impact of ensemble learning in sensitive healthcare scenarios.

---

## 📌 Project Overview

Using the **Breast Cancer Wisconsin Diagnostic Dataset**, the goal is to classify tumors as **malignant (M)** or **benign (B)** using over 30 numerical diagnostic features. The project involves data cleaning, exploratory data analysis, model building using advanced ensemble techniques, and comparison of results.

---

## 🎯 Objectives

- Load and clean the medical dataset
- Explore class-wise distributions of diagnostic metrics
- Train two state-of-the-art ensemble models: **XGBoost** and **LightGBM**
- Evaluate models using key classification metrics
- Compare model performance and generalization

---

## 🗂️ Dataset Details

- Source: `Breast Cancer Wisconsin (Diagnostic)` dataset  
- Features: 30 numerical columns (mean radius, texture, concavity, etc.)
- Target: Diagnosis (M = Malignant, B = Benign)

---

## 🔍 Key Highlights

### ✅ Data Loading & Preprocessing

- Loaded the dataset using Pandas  
- Dropped unnecessary columns (`id`, `Unnamed: 32`)  
- Mapped categorical target labels (M → 1, B → 0)  
- Verified data integrity (no nulls, correct formats)

### 📊 Exploratory Data Analysis (EDA)

- Used a custom **Dis_Stats** module for in-depth feature stats  
- Visualized class distributions via:
  - Displots – feature spread
  - Violin plots – separability
  - KDE plots – density overlap
- Compared means, variances, and ranges between benign and malignant samples

### 🤖 Model Building

- Trained two gradient boosting models:
  - **XGBoost Classifier** – regularized boosting with depth control
  - **LightGBM Classifier** – fast, leaf-wise gradient boosting
- Used an **80-20 train-test split** for fair model evaluation

### 📈 Model Evaluation

- Metrics used:
  - **Accuracy Score**
  - **Precision, Recall, F1-Score** (from `classification_report`)
- Compared training vs testing performance to detect overfitting
- Observed strong and balanced results on both models

> ✅ *LightGBM slightly outperformed XGBoost on the test set, with faster training and lower overfitting risk.*

---

## 🛠 Tools & Technologies Used

- **Python Libraries:**
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – EDA & visualization
  - `scikit-learn` – metrics and splitting
  - `xgboost`, `lightgbm` – model implementation
- **Custom Module:**
  - `Dis_Stats` – custom statistical EDA functions
- **Platform:** Jupyter Notebook

---

## 📈 Results Summary

| Model     | Accuracy | F1-Score | Observations |
|-----------|----------|----------|--------------|
| XGBoost   | ~97%     | High     | Strong, slight overfit |
| LightGBM  | ~98%     | Higher   | Better generalization, faster training |

---

## 🌐 Deployment

The project has been deployed as an **interactive Streamlit web application** for real-time breast cancer prediction. Users can input diagnostic feature values and instantly receive predictions from both **XGBoost** and **LightGBM** models.

Live Demo : https://breast-cancer-web-application.streamlit.app/
