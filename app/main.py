from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Credit Risk Prediction API")

model = joblib.load("models/credit_risk_xgboost.pkl")


class BorrowerInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: float


@app.get("/")
def root():
    return {"message": "Credit Risk API is running"}


@app.post("/predict")
def predict(input_data: BorrowerInput):
    df = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": input_data.RevolvingUtilizationOfUnsecuredLines,
        "age": input_data.age,
        "NumberOfTime30-59DaysPastDueNotWorse": input_data.NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": input_data.DebtRatio,
        "MonthlyIncome": input_data.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": input_data.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": input_data.NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": input_data.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": input_data.NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": input_data.NumberOfDependents
    }])

    pred_class = int(model.predict(df)[0])
    pred_prob = float(model.predict_proba(df)[0][1])

    return {
        "predicted_class": pred_class,
        "default_probability": round(pred_prob, 4)
    }

