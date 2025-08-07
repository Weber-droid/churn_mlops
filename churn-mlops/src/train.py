# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def preprocess(df):
    df.dropna(inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(['customerID'], axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df)
    return df

def train():
    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "churn_model")

        print(f"Model accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
