from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# def test_health():
#     """
#     teste le endpoint '/statut'
#     """
#     response = client.get("/statut")
#     assert response.status_code == 200
#     assert response.json() == {
#         "status": "healthy"
#     }


def test_root():
    """
    teste le endpoint '/'
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"


def test_predict_setosa():
    """
    teste la prédiction setosa
    """
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "setosa"



def test_predict_virginica():
    """
    Teste la prédiction virginica
    """
    data = {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == "virginica"


def test_predict_invalid_value():
    """
    teste l'invalidité des données
    """
    data = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 422



def test_predict_missing_value():
    """
    teste une prédiction avec une données manquante
    """
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post(
        "/predict",
        json=data
    )

    assert response.status_code == 422