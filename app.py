import numpy as np
import pandas as pd
import pickle

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [1 if y > 0.5 else 0 for y in y_pred]
        return class_pred


filename = 'finalized_model.sav'


model = pickle.load(open(filename, 'rb'))


app = FastAPI()


class DiabetesInput(BaseModel):
    Pregnancies: list[int]
    Glucose: list[int]
    BloodPressure: list[int]
    SkinThickness: list[int]
    Insulin: list[int]
    BMI: list[int]
    DiabetesPedigreeFunction: list[float]
    Age: list[int]


@app.get("/")
def send_props():
    return HTMLResponse(content=open("index.html").read(), status_code=200)


@app.post("/submit")
def update_item(data: DiabetesInput):
    new_inputs = pd.DataFrame({
        'Pregnancies': data.Pregnancies,
        'Glucose': data.Glucose,
        'BloodPressure': data.BloodPressure,
        'SkinThickness': data.SkinThickness,
        'Insulin': data.Insulin,
        'BMI': data.BMI,
        'DiabetesPedigreeFunction': data.DiabetesPedigreeFunction,
        'Age': data.Age
    })

    return model.predict(new_inputs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
