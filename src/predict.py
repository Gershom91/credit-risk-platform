import pandas as pd
import joblib


def load_model(path: str):
    return joblib.load(path)


def make_sample_input() -> pd.DataFrame:
    sample_data = pd.DataFrame([
        {
            "RevolvingUtilizationOfUnsecuredLines": 0.85,
            "age": 45,
            "NumberOfTime30-59DaysPastDueNotWorse": 2,
            "DebtRatio": 0.65,
            "MonthlyIncome": 5000,
            "NumberOfOpenCreditLinesAndLoans": 8,
            "NumberOfTimes90DaysLate": 1,
            "NumberRealEstateLoansOrLines": 1,
            "NumberOfTime60-89DaysPastDueNotWorse": 1,
            "NumberOfDependents": 2
        }
    ])
    return sample_data


def predict_risk(model, input_df: pd.DataFrame):
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1]
    return pred_class, pred_prob


if __name__ == "__main__":
    model = load_model("models/credit_risk_xgboost.pkl")
    sample_input = make_sample_input()

    pred_class, pred_prob = predict_risk(model, sample_input)

    print("Sample borrower data:")
    print(sample_input)

    print("\nPrediction:")
    print("Predicted class:", pred_class)
    print("Default probability:", round(pred_prob, 4))
