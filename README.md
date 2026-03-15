# Credit Risk Decision Platform

## Overview
An end-to-end machine learning system for predicting borrower default risk using structured financial data from the Kaggle **Give Me Some Credit** dataset.

This project covers:
- data cleaning and preprocessing
- feature engineering and outlier handling
- model training and evaluation
- threshold tuning
- feature importance analysis
- local prediction script
- FastAPI deployment layer

## Dataset
- Source: Kaggle - Give Me Some Credit
- Files used:
  - `cs-training.csv`
  - `cs-test.csv`

## Project Structure
```text
credit-risk-platform/
│
├── app/
│   └── main.py
├── data/
│   ├── processed/
│   └── raw/
│       ├── cs-training.csv
│       └── cs-test.csv
├── models/
│   └── credit_risk_xgboost.pkl
├── notebooks/
│   └── 01_data_understanding.ipynb
├── reports/
│   └── feature_importance.png
├── sql/
├── src/
│   ├── predict.py
│   └── train_model.py
├── requirements.txt
└── README.md

## System Architecture

```text
Data (Kaggle Credit Dataset)
        ↓
Data Cleaning & Feature Engineering
        ↓
Model Training (XGBoost)
        ↓
Saved Model (.pkl)
        ↓
Prediction Script
        ↓
FastAPI Endpoint
        ↓
Client Request → Default Risk Prediction

## Model Performance

| Model | ROC-AUC | Recall | F1 Score |
|------|------|------|------|
| Logistic Regression | 0.83 | 0.74 | 0.29 |
| Random Forest | 0.84 | 0.15 | 0.24 |
| XGBoost | 0.87 | 0.69 | 0.38 |

### Final Model Selection

XGBoost was selected as the final model because it achieved the best overall ROC-AUC while maintaining strong recall for detecting high-risk borrowers.