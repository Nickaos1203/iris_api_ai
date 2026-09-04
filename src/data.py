from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_and_prepare_data():
    """
    Charge et prépare le dataset Iris.
    """

    iris = load_iris()

    X = iris.data
    y = iris.target

    # Validation de la structure
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Le nombre d'observations et de labels est différent."
        )

    # Validation des valeurs manquantes
    if X is None or y is None:
        raise ValueError("Les données sont absentes.")

    # Vérification du nombre de classes
    if len(set(y)) != 3:
        raise ValueError(
            "Le dataset doit contenir exactement 3 classes."
        )

    # Séparation train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test