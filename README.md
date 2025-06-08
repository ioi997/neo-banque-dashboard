# 🏦 Dashboard IA – Néo-Banque

Ce projet met en œuvre une application complète composée :

- d'une **API FastAPI** pour évaluer l’éligibilité d’un client à un prêt,
- d’un **dashboard Streamlit** à destination des conseillers bancaires,
- d’un modèle de scoring utilisant `RandomForestClassifier` avec SHAP pour l’explicabilité.

---

## 📌 Objectifs du projet

Déployé dans le cadre d’un cas d’étude en Mastère IA, ce projet vise à :

- Prédire l’éligibilité d’un client à un prêt via un score de probabilité
- Fournir une **explication intelligible** des facteurs influençant la décision (SHAP)
- Permettre au conseiller bancaire d'accéder aux informations clés via un **dashboard clair**

---

## 🔧 Technologies utilisées

- 🐍 Python
- ⚡ FastAPI (API backend)
- 📈 Streamlit (dashboard)
- 🤖 Scikit-learn (modèle ML)
- 📊 SHAP (explicabilité)
- ☁️ Render.com (déploiement cloud)

---
🌐 Accéder aux applications déployées
| Composant    | URL                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| 🔗 Dashboard | [https://neo-banque-dashboard.onrender.com](https://neo-banque-dashboard.onrender.com)                 |
| 🔗 API       | [https://neo-banque-dashboard.onrender.com/predict](https://neo-banque-dashboard.onrender.com/predict) |

 ---
📂 Structure du projet
neo-banque-dashboard/

├── app/
│   ├── model.pkl             # Modèle ML entraîné (RandomForest)
│   └── explainer.pkl         # Explainer SHAP
├── data/
│   └── clients.csv           # Données des clients à scorer
├── api.py                    # API FastAPI
├── streamlit_app.py          # Interface utilisateur
├── requirements.txt
└── README.md

 ---
 🔐 RGPD & Éthique
Aucune donnée personnelle (nom, adresse, email) n’est collectée.
Seules les données financières sont utilisées à des fins de scoring.
Le modèle est explicable et transparent via SHAP (compliant IA éthique).
