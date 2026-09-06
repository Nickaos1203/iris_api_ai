# Import des bibliothèques Python
import os
import json
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from datetime import datetime


# ==========================================
# Variables de configuration
# ==========================================

# Chemin d'enregistrement du modèle entrainé
MODEL_PATH = "models/iris_model.joblib"

# Chemin d'enregistrement des métriques d'entrainement
METRICS_PATH = "models/metrics.json"

# critères d'évaluation du modèle
MIN_SCORE = 0.90
RANDOM_STATE = 42


# ==========================================
# 1. Préparation des données
# ==========================================

def load_and_prepare_data():
    """Charge les données du dataset Iris et prépare les données

    Returns:
        X_train: liste des features d'entrainement
        X_test: liste des features de test
        y_train: liste des targets d'entrainement
        y_test: liste des targets de test
    """

    # Chargement du dataset Iris
    iris = load_iris()

    # Séparation features/target
    X = iris.data
    y = iris.target

    # Vérification de la structure
    if X.shape[1] != 4:
        raise ValueError(
            "Le dataset doit contenir 4 variables explicatives."
        )

    # Vérification des classes
    if len(set(y)) != 3:
        raise ValueError(
            "Le dataset doit contenir 3 classes."
        )

    # Vérification des valeurs manquantes
    if hasattr(X, "any"):
        if not X.all():
            raise ValueError(
                "Les données contiennent des valeurs invalides."
            )

    # Séparation entraînement / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test


# ==========================================
# 2. Entraînement du modèle
# ==========================================

def train_model(X_train, y_train):
    """
    Entraine le modèle de machine learning

    Args:
        X_train (list(float)): liste des features d'entrainement
        y_train (_type_): liste des targets d'entrainement

    Returns:
        sklearn_model: modèle entrainé
    """

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model


# ==========================================
# 3. Évaluation du modèle
# ==========================================

def evaluate_model(X_test, y_test):
    """évalue le modèle

    Args:
        X_test: liste des features de test
        y_test: liste des targets de test

    Returns:
        dict: dictionnaire comprenant les métriques calculées
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    # Chargement du modèle
    model = joblib.load(MODEL_PATH)

    # Prédictions
    predictions = model.predict(X_test)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    # Affichage
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nRapport de classification :")
    print(classification_report(y_test, predictions))

    # Métriques
    metrics = {
        "model": type(model).__name__,
        "n_estimators": model.n_estimators,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1-score": round(f1, 4),
        "date": datetime.now().isoformat(timespec="seconds")
    }

    # Contrôle des seuils
    metrics_to_check = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1
    }

    for metric_name, value in metrics_to_check.items():

        if value < MIN_SCORE:
            print(
                f"Modèle rejeté : {metric_name}={value:.4f} "
                f"< seuil={MIN_SCORE:.2f}"
            )
            raise SystemExit(1)

    print(
        f"\nModèle validé : toutes les métriques sont "
        f">= seuil={MIN_SCORE:.2f}"
    )

    # Retourne les métriques
    return metrics


# ==========================================
# 4. Sauvegarde du modèle
# ==========================================

def save_model(model, metrics):
    """_summary_

    Args:
        model (_type_): _description_
        metrics (_type_): _description_
    """

    os.makedirs("models", exist_ok=True)

    # Sauvegarde du modèle
    joblib.dump(model, MODEL_PATH)

    # Sauvegarde des métriques
    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"Modèle sauvegardé : {MODEL_PATH}")
    print(f"Métriques sauvegardées : {METRICS_PATH}")


# ==========================================
# 5. Pipeline complète
# ==========================================

def main():
    print("Début de la pipeline PREPARATION - ENTRAINEMENT - ENREGISTREMENT")
    # Préparation des données
    print("\n[1/4] Préparation des données...")

    X_train, X_test, y_train, y_test = (load_and_prepare_data())

    print(f"Train : {len(X_train)} observations")
    print(f"Test  : {len(X_test)} observations")


    # Entraînement
    print("\n[2/4] Entraînement du modèle...")

    model = train_model(X_train, y_train)

    print("Type du modèle :")
    print(type(model))

    # Évaluation
    print("\n[3/4] Évaluation du modèle...")

    metrics = evaluate_model(X_test, y_test)

    # Sauvegarde
    print("\n[4/4] Sauvegarde du modèle...")

    save_model(model, metrics)

    print("Fin de la pipeline PREPARATION - ENTRAINEMENT - ENREGISTREMENT")


if __name__ == "__main__":
    main()