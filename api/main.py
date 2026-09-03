from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np


app = FastAPI(
    title="Iris Classification API",
    description="API de prédiction du modèle Iris",
    version="1.0.0"
)


# Chargement du modèle
model = joblib.load("models/iris_model.joblib")


# Classes Iris
target_names = [
    "setosa",
    "versicolor",
    "virginica"
]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


@app.get("/")
def root():
    return {
        "message": "Iris Classification API",
        "status": "running"
    }


@app.get("/statut")
def statut():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: IrisFeatures):

    features = np.array([
        [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]
    ])

    prediction = model.predict(features)

    predicted_class = target_names[prediction[0]]

    return {
        "prediction": predicted_class
    }