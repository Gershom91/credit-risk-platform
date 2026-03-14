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
