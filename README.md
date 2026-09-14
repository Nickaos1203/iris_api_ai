# 🌸 IRIS API AI


## 📌 Présentation

**IRIS API AI** est un projet de Machine Learning et de MLOps basé sur le dataset **Iris**.

L'objectif est de construire une chaîne complète allant de l'entraînement et de la validation d'un modèle de Machine Learning jusqu'à son exposition sous forme d'API, son utilisation via une interface graphique et son monitoring.

Le projet met en œuvre :

- un modèle `RandomForestClassifier` entraîné avec **scikit-learn** ;
- une API REST développée avec **FastAPI** ;
- une interface utilisateur développée avec **Streamlit** ;
- la conteneurisation avec **Docker** et **Docker Compose** ;
- la supervision avec **Prometheus** et **Grafana** ;
- des tests automatisés avec **pytest** ;
- une pipeline **CI/CD GitHub Actions** ;
- le déploiement sur **Render**.

Le projet constitue ainsi un exemple complet de mise en production d'un modèle de Machine Learning dans une démarche MLOps.

---

## 🎯 Objectifs du projet

Le projet répond aux objectifs suivants :

1. Préparer les données du dataset Iris.
2. Séparer les données en jeux d'entraînement et de test.
3. Entraîner un modèle de classification.
4. Évaluer automatiquement ses performances.
5. Refuser le modèle si ses performances sont insuffisantes.
6. Sauvegarder le modèle entraîné.
7. Exposer le modèle via une API REST.
8. Valider les données reçues par l'API.
9. Fournir une interface utilisateur simple pour effectuer des prédictions.
10. Instrumenter l'API avec des métriques Prometheus.
11. Visualiser les métriques avec Grafana.
12. Automatiser les tests et contrôles avec GitHub Actions.
13. Construire et tester les images Docker.
14. Déployer automatiquement l'application sur Render après validation de la branche `main`.

---

## 🧠 Dataset et modèle

Le projet utilise le dataset **Iris** fourni directement par `scikit-learn`.

Le dataset contient :

- **150 observations** ;
- **4 variables explicatives** :
  - longueur du sépale ;
  - largeur du sépale ;
  - longueur du pétale ;
  - largeur du pétale ;
- **3 classes** :
  - `setosa` ;
  - `versicolor` ;
  - `virginica`.

### Modèle

Le modèle utilisé est un :

```text
RandomForestClassifier
```

avec :

```text
n_estimators = 100
random_state = 42
```

La séparation des données utilise :

```text
80 % entraînement
20 % test
```

avec une stratification des classes.

### Seuil de qualité

Le pipeline impose un seuil minimal de **0,88** pour chacune des métriques suivantes :

- Accuracy
- Precision
- Recall
- F1-score

Si l'une des métriques est inférieure à `0.88`, le modèle est rejeté et la pipeline CI/CD échoue.

---

## 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │      Dataset Iris   │
                         │    scikit-learn     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Pipeline ML         │
                         │ préparation        │
                         │ entraînement        │
                         │ évaluation          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Modèle validé       │
                         │ Random Forest       │
                         │ iris_model.joblib   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     FastAPI         │
                         │      /predict       │
                         │      /metrics/      │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌──────────────────┐
          │   Streamlit     │             │   Prometheus     │
          │ Interface Web   │             │   Monitoring     │
          └─────────────────┘             └────────┬─────────┘
                                                   │
                                                   ▼
                                           ┌──────────────────┐
                                           │     Grafana      │
                                           │   Dashboard      │
                                           └──────────────────┘

                  GitHub Actions
                         │
                         ▼
        ┌────────────────────────────────────┐
        │ CI/CD                              │
        │ 1. Préparation                     │
        │ 2. Validation du modèle            │
        │ 3. Tests pytest                    │
        │ 4. Test Docker développement       │
        │ 5. Test Docker production          │
        │ 6. Déploiement Render              │
        └────────────────────────────────────┘
```

---

## 📁 Structure du projet

```text
iris_api_ai/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── api/
│   └── main.py
│
├── data/
│   └── ...
│
├── models/
│   ├── iris_model.joblib
│   └── metrics.json
│
├── notebooks/
│   └── ...
│
├── prometheus/
│   └── prometheus.yml
│
├── src/
│   └── pipeline_model.py
│
├── streamlit/
│   └── app.py
│
├── tests/
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile.api
├── Dockerfile.api.render
├── Dockerfile.grafana.render
├── Dockerfile.prometheus.render
├── Dockerfile.streamlit
├── Dockerfile.streamlit.render
├── docker-compose.yml
├── docker-compose.render.yml
├── requirements.txt
├── requirements_dev.txt
└── README.md
```

---

## ⚙️ Prérequis

Pour exécuter le projet localement sans Docker :

- Python **3.12**
- pip
- Git

Pour exécuter l'ensemble de la stack :

- Docker
- Docker Compose

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Nickaos1203/iris_api_ai.git
cd iris_api_ai
```

### 2. Créer un environnement virtuel

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pour installer également les dépendances de développement :

```bash
pip install -r requirements_dev.txt
```

---

# 🤖 Pipeline Machine Learning

La pipeline Machine Learning est disponible dans :

```text
src/pipeline_model.py
```

Elle réalise quatre étapes principales :

```text
1. Préparation des données
        ↓
2. Entraînement
        ↓
3. Évaluation
        ↓
4. Sauvegarde
```

Pour lancer la pipeline :

```bash
python src/pipeline_model.py
```

Le modèle est sauvegardé dans :

```text
models/iris_model.joblib
```

Les métriques sont sauvegardées dans :

```text
models/metrics.json
```

### Exemple de métriques

Le fichier `metrics.json` contient notamment :

```json
{
    "model": "RandomForestClassifier",
    "n_estimators": 100,
    "accuracy": 0.96,
    "precision": 0.96,
    "recall": 0.96,
    "f1-score": 0.96
}
```

Les valeurs exactes dépendent de l'exécution de la pipeline.

---

# 🚀 API FastAPI

L'API est située dans :

```text
api/main.py
```

Elle charge automatiquement le modèle :

```text
models/iris_model.joblib
```

## Démarrer l'API

```bash
cd api
uvicorn main:app --reload
```

L'API est alors disponible à :

```text
http://127.0.0.1:8000
```

Documentation interactive Swagger :

```text
http://127.0.0.1:8000/docs
```

---

## 🔌 Endpoints

### `GET /`

Vérifie que l'API fonctionne.

```bash
curl http://127.0.0.1:8000/
```

Réponse :

```json
{
    "message": "Bienvenue sur IRIS API AI",
    "status": "running"
}
```

### `POST /predict`

Effectue une prédiction.

Exemple :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

Réponse :

```json
{
    "prediction": "setosa"
}
```

Les quatre variables sont obligatoires et doivent être strictement positives.

### `GET /metrics/`

Expose les métriques de l'API au format Prometheus.

```bash
curl http://127.0.0.1:8000/metrics/
```

---

# 🖥️ Interface Streamlit

L'application Streamlit fournit une interface graphique permettant de saisir les caractéristiques d'une fleur et d'appeler l'API FastAPI.

Pour la lancer :

```bash
streamlit run streamlit/app.py
```

L'interface est disponible par défaut à :

```text
http://localhost:8501
```

L'URL de l'API peut être configurée avec la variable d'environnement :

```text
API_URL
```

Exemple :

```bash
API_URL=http://127.0.0.1:8000 streamlit run streamlit/app.py
```

---

# 🐳 Docker

Le projet fournit plusieurs Dockerfiles afin de séparer les différents composants.

## Docker Compose

La stack locale comprend :

- FastAPI
- Streamlit
- Prometheus
- Grafana

Lancer l'ensemble :

```bash
docker compose up --build
```

### Services

| Service | Port | Rôle |
|---|---:|---|
| FastAPI | `8000` | API de prédiction |
| Streamlit | `8501` | Interface utilisateur |
| Prometheus | `9090` | Collecte des métriques |
| Grafana | `3000` | Visualisation |

### Arrêter les services

```bash
docker compose down
```

### Voir les conteneurs

```bash
docker compose ps
```

### Voir les logs

```bash
docker compose logs -f
```

---

# 📊 Monitoring avec Prometheus et Grafana

Le monitoring permet de suivre le comportement de l'API en production.

Prometheus collecte les métriques exposées par :

```text
http://api:8000/metrics/
```

La configuration se trouve dans :

```text
prometheus/prometheus.yml
```

L'intervalle de collecte est de :

```text
5 secondes
```

## Métriques principales

### Disponibilité

```text
iris_api_up
```

Indique si l'API est disponible.

### Nombre de requêtes

```text
iris_api_requests_total
```

Compte le nombre total de requêtes reçues.

### Nombre de prédictions

```text
iris_predictions_total
```

Compte les prédictions réalisées et permet de les répartir par espèce.

### Nombre d'erreurs

```text
iris_api_errors_total
```

Compte les erreurs rencontrées par l'API.

### Temps de réponse

```text
iris_api_request_duration_seconds
```

Mesure la durée des requêtes.

---

## Accès à Grafana

Une fois la stack lancée :

```text
http://localhost:3000
```

Prometheus :

```text
http://localhost:9090
```

L'objectif du monitoring est de permettre le suivi de :

- la disponibilité de l'API ;
- la volumétrie des requêtes ;
- le nombre de prédictions ;
- la répartition des prédictions par espèce ;
- les erreurs ;
- les temps de réponse.

---

# 🧪 Tests automatisés

Les tests sont situés dans :

```text
tests/
```

Ils vérifient notamment :

- le fonctionnement de l'endpoint `/` ;
- une prédiction `setosa` ;
- une prédiction `virginica` ;
- le rejet d'une valeur négative ;
- le rejet d'une donnée manquante.

Lancer les tests :

```bash
pytest -v
```

---

# 🔄 CI/CD avec GitHub Actions

Le workflow CI/CD est défini dans :

```text
.github/workflows/ci-cd.yml
```

La pipeline est déclenchée :

- lors d'un `push` sur `main` ;
- lors de la création ou modification d'une Pull Request vers `main`.

## Étapes de la pipeline

```text
Push / Pull Request
        │
        ▼
┌─────────────────────────────┐
│ 1. Préparation              │
│ Python 3.12                 │
│ Installation dépendances    │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 2. Validation du modèle     │
│ pipeline_model.py           │
│ seuil qualité >= 0.88       │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 3. Tests automatisés        │
│ pytest -v                   │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 4. Docker développement     │
│ build + démarrage + /metrics│
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 5. Docker production        │
│ docker-compose.render.yml   │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 6. Déploiement Render       │
│ uniquement sur main         │
└─────────────────────────────┘
```

Cette organisation permet de bloquer un déploiement lorsqu'une étape critique échoue.

---

# ☁️ Déploiement sur Render

Le projet contient une configuration spécifique pour l'environnement Render :

```text
docker-compose.render.yml
```

ainsi que plusieurs Dockerfiles dédiés :

```text
Dockerfile.api.render
Dockerfile.streamlit.render
Dockerfile.prometheus.render
Dockerfile.grafana.render
```

Le workflow GitHub Actions déclenche le déploiement uniquement après validation des étapes précédentes et uniquement lors d'un push sur `main`.

Les services déployés sont :

- API FastAPI ;
- application Streamlit ;
- Prometheus ;
- Grafana.

## Secrets GitHub

Le workflow utilise les secrets GitHub suivants :

```text
RENDER_API_DEPLOY_HOOK
RENDER_STREAMLIT_DEPLOY_HOOK
RENDER_PROMETHEUS_DEPLOY_HOOK
RENDER_GRAFANA_DEPLOY_HOOK
```

Ces valeurs ne doivent **jamais être stockées directement dans le dépôt Git**.

Elles doivent être configurées dans :

```text
GitHub
→ Settings
→ Secrets and variables
→ Actions
```

---

# 🔐 Bonnes pratiques de sécurité

Le projet applique plusieurs principes de base :

- aucun secret de déploiement dans le code source ;
- utilisation des GitHub Actions Secrets pour les hooks Render ;
- validation des données entrantes avec Pydantic ;
- séparation des environnements Docker ;
- validation du modèle avant son utilisation ;
- tests automatisés avant le déploiement.

Les fichiers `.env`, environnements virtuels et fichiers temporaires doivent rester exclus du dépôt via `.gitignore`.

---

# 📈 Démarche MLOps

Le projet couvre les principales étapes d'une chaîne MLOps :

| Étape | Implémentation |
|---|---|
| Données | Dataset Iris `scikit-learn` |
| Préparation | `src/pipeline_model.py` |
| Entraînement | Random Forest |
| Évaluation | Accuracy, Precision, Recall, F1 |
| Quality Gate | seuil minimal de `0.88` |
| Packaging modèle | `joblib` |
| Serving | FastAPI |
| Interface | Streamlit |
| Tests | pytest |
| Conteneurisation | Docker |
| Orchestration locale | Docker Compose |
| Monitoring | Prometheus |
| Visualisation | Grafana |
| CI | GitHub Actions |
| CD | Render |
| Secrets | GitHub Actions Secrets |

---

# 🧭 Reproduire le projet de bout en bout

Pour reproduire le projet localement :

### Étape 1 — Cloner

```bash
git clone https://github.com/Nickaos1203/iris_api_ai.git
cd iris_api_ai
```

### Étape 2 — Installer les dépendances

```bash
python -m venv .venv
```

Windows :

```bash
.venv\Scripts\activate
```

Linux/macOS :

```bash
source .venv/bin/activate
```

Puis :

```bash
pip install -r requirements.txt
```

### Étape 3 — Entraîner et valider le modèle

```bash
python src/pipeline_model.py
```

### Étape 4 — Lancer les tests

```bash
pytest -v
```

### Étape 5 — Démarrer l'application complète

```bash
docker compose up --build
```

### Étape 6 — Vérifier les services

API :

```text
http://localhost:8000
```

Swagger :

```text
http://localhost:8000/docs
```

Streamlit :

```text
http://localhost:8501
```

Prometheus :

```text
http://localhost:9090
```

Grafana :

```text
http://localhost:3000
```

---

# 🛠️ Technologies utilisées

- **Python 3.12**
- **scikit-learn**
- **Pandas**
- **Joblib**
- **FastAPI**
- **Pydantic**
- **Uvicorn**
- **Streamlit**
- **pytest**
- **Docker**
- **Docker Compose**
- **Prometheus**
- **Grafana**
- **GitHub Actions**
- **Render**

---

# 📚 Compétences mises en œuvre

Ce projet permet de mettre en pratique plusieurs compétences :

### Machine Learning

- préparation des données ;
- entraînement d'un modèle de classification ;
- évaluation ;
- contrôle de la qualité d'un modèle ;
- sérialisation du modèle.

### Développement API

- conception d'une API REST ;
- validation des entrées ;
- documentation Swagger ;
- gestion des erreurs ;
- exposition d'un modèle ML.

### MLOps

- reproductibilité de l'entraînement ;
- validation automatique du modèle ;
- tests automatisés ;
- packaging ;
- conteneurisation ;
- monitoring ;
- CI/CD.

### DevOps

- Docker ;
- Docker Compose ;
- GitHub Actions ;
- gestion des secrets ;
- déploiement automatisé.

---

# 🔗 Dépôt GitHub

Le projet est disponible sur :

https://github.com/Nickaos1203/iris_api_ai

---

# 👤 Auteur

Projet **IRIS API AI** réalisé dans le cadre d'une démarche de développement **Machine Learning / MLOps / CI-CD**.

---

## 📄 Licence

Projet destiné à un usage pédagogique et de démonstration.
