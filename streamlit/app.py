import streamlit as st
import requests


# Configuration de la page
st.set_page_config(
    page_title="Iris MLOps AI",
    page_icon="🌸",
    layout="centered"
)


# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000"


# Titre
st.title("🌸 Iris MLOps AI")

st.write(
    "Cette application permet de prédire l'espèce d'une fleur "
    "à partir de la taille de ses pétales et de ses sépales."
)


# Séparation visuelle
st.divider()


# Formulaire
st.subheader("Caractéristiques de la fleur")

sepal_length = st.number_input(
    "Longueur du sépale (cm)",
    min_value=0.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Largeur du sépale (cm)",
    min_value=0.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Longueur du pétale (cm)",
    min_value=0.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Largeur du pétale (cm)",
    min_value=0.0,
    value=0.2,
    step=0.1
)


# Bouton de prédiction
if st.button("🔍 Prédire l'espèce", use_container_width=True):

    # Données envoyées à l'API
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:

        # Appel de l'API FastAPI
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=5
        )

        # Vérification de la réponse
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            st.success(
                f"🌸 Espèce prédite : **Iris {prediction}**"
            )
        else:
            st.error(
                f"Erreur API : {response.status_code}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Impossible de contacter l'API FastAPI. "
            "Vérifiez que l'API est démarrée."
        )

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ L'API met trop de temps à répondre."
        )

    except Exception as e:

        st.error(
            f"Une erreur est survenue : {e}"
        )


# Informations sur l'API
st.divider()

st.caption(
    f"API de prédiction : {API_URL}"
)