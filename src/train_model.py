import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df[df["age"] > 0]

    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=1)

    df["DebtRatio"] = df["DebtRatio"].clip(
        upper=df["DebtRatio"].quantile(0.99)
    )

    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(
        df["MonthlyIncome"].median()
    )
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
        df["NumberOfDependents"].median()
    )

    return df


def split_data(df: pd.DataFrame):
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def train_xgb_model(X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    df = load_data("data/raw/cs-training.csv")
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_xgb_model(X_train, y_train)

    joblib.dump(model, "models/credit_risk_xgboost.pkl")
    print("Model trained and saved successfully.")
    