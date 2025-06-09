#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import requests
import os
import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Dashboard Néo-Banque", layout="centered")

# Ajout de la section RGPD dans la sidebar
with st.sidebar.expander("🔐 Données & RGPD"):
    st.markdown("""
    **Conformité RGPD**

    - Ce dashboard traite des données **pseudonymisées**
    - Aucune donnée personnelle (nom, email...) n’est utilisée
    - Les données sont utilisées uniquement à des fins de **scoring de prêt**
    - Le traitement est **explicable** grâce aux outils SHAP

    👉 Ce traitement respecte les principes du RGPD :
    - Licéité, transparence, finalité, minimisation
    - Pas de stockage ni de profilage automatisé externe
    """)

clients = pd.read_csv("data/clients.csv").reset_index().rename(columns={"index": "id"})

st.title("📊 Dashboard conseiller")
selected_id = st.selectbox("Choisir un client", clients.index)

client = clients.loc[selected_id]
st.subheader("Informations client")
st.write(client)

API_URL = os.getenv("API_URL", "https://neo-api-jigt.onrender.com/predict")

if st.button("📤 Envoyer pour scoring"):
    input_data = {
        "age": int(client["age"]),
        "revenu": float(client["revenu"]),
        "anciennete": int(client["anciennete"]),
        "nb_incidents": int(client["nb_incidents"]),
        "score_credit": float(client["score_credit"]),
    }

    try:
        logging.info(f"Envoi de la requête à l'API : {input_data}")
        res = requests.post(API_URL, json=input_data, timeout=10)
        res.raise_for_status()  # Lève une exception pour les codes HTTP d'erreur
        response_data = res.json()
        score = response_data["score"]
        explanations_from_api = response_data.get("explanations", [])

        # Nouvelle section avec jauge visuelle
        st.subheader("Score d'éligibilité au prêt")
        
        # Jauge colorée avec indicateur
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            # Barre de progression colorée
            progress_value = score
            progress_color = "red" if score < 0.3 else "orange" if score < 0.5 else "green"
            
            st.markdown(f"""
            <div style="margin-bottom: 10px; text-align: center; font-weight: bold; font-size: 20px;">
                {score * 100:.1f}%
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(progress_value)
            
            # Ajout d'indicateurs sous la jauge
            st.markdown("""
            <div style="display: flex; justify-content: space-between; margin-top: -10px;">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: -15px;">
                <span>Risque élevé</span>
                <span>Limite</span>
                <span>Très favorable</span>
            </div>
            """, unsafe_allow_html=True)

        # Message d'interprétation
        if score > 0.5:
            st.success("✅ Client éligible probable au prêt.")
        elif score > 0.3:
            st.warning("⚠️ Client à risque modéré - Analyse approfondie recommandée.")
        else:
            st.error("❌ Client à haut risque - Probablement inéligible.")

        st.subheader("Comprendre le score (facteurs clés)")
        if explanations_from_api:
            for explanation_text in explanations_from_api:
                if "positivement" in explanation_text:
                    st.write(f"⬆️ {explanation_text}")
                elif "négativement" in explanation_text:
                    st.write(f"⬇️ {explanation_text}")
                else:
                    st.write(f"➡️ {explanation_text}")
        else:
            st.info("Aucune explication détaillée disponible pour le moment ou une erreur s'est produite côté API.")

    except requests.exceptions.ConnectionError:
        st.error(f"Erreur de connexion à l'API. Vérifiez que l'API est accessible à l'adresse {API_URL}.")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la communication avec l'API : {e}. Vérifiez l'URL et la configuration de l'API.")
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite : {e}")
