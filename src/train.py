import os
import joblib

from sklearn.ensemble import RandomForestClassifier

from src.data import load_and_prepare_data


MODEL_PATH = "models/iris_model.joblib"


def train_model():

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Modèle enregistré : {MODEL_PATH}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()