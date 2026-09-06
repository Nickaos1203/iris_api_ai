from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import time

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    make_asgi_app
)


app = FastAPI(
    title="Iris API AI",
    description="API de prédiction du modèle Iris",
    version="1.0.0"
)


# ============================================================
# Chargement du modèle
# ============================================================

model = joblib.load("models/iris_model.joblib")


target_names = [
    "setosa",
    "versicolor",
    "virginica"
]


# ============================================================
# Métriques Prometheus
# ============================================================

REQUEST_COUNT = Counter(
    "iris_api_requests_total",
    "Nombre total de requêtes reçues par l'API"
)


PREDICTION_COUNT = Counter(
    "iris_predictions_total",
    "Nombre total de prédictions réalisées",
    ["species"]
)


ERROR_COUNT = Counter(
    "iris_api_errors_total",
    "Nombre total d'erreurs de l'API"
)


REQUEST_LATENCY = Histogram(
    "iris_api_request_duration_seconds",
    "Durée des requêtes API en secondes"
)


API_UP = Gauge(
    "iris_api_up",
    "État de disponibilité de l'API"
)

API_UP.set(1)


# ============================================================
# Endpoint Prometheus
# ============================================================

metrics_app = make_asgi_app()

app.mount(
    "/metrics",
    metrics_app
)


# ============================================================
# Modèle de données
# ============================================================

class IrisFeatures(BaseModel):

    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():

    REQUEST_COUNT.inc()

    return {
        "message": "Bienvenue sur IRIS API AI",
        "status": "running"
    }



@app.post("/predict")
def predict(data: IrisFeatures):

    start_time = time.time()

    REQUEST_COUNT.inc()

    try:

        features = pd.DataFrame([{
            "sepal length (cm)": data.sepal_length,
            "sepal width (cm)": data.sepal_width,
            "petal length (cm)": data.petal_length,
            "petal width (cm)": data.petal_width
        }])

        prediction = model.predict(features)

        predicted_class = target_names[prediction[0]]

        # Compteur par espèce
        PREDICTION_COUNT.labels(
            species=predicted_class
        ).inc()

        return {
            "prediction": predicted_class
        }

    except Exception:

        ERROR_COUNT.inc()

        raise

    finally:

        duration = time.time() - start_time

        REQUEST_LATENCY.observe(duration)