#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Dashboard Néo-Banque", layout="centered")

clients = pd.read_csv("data/clients.csv").reset_index().rename(columns={"index": "id"})

st.title("📊 Dashboard conseiller")
selected_id = st.selectbox("Choisir un client", clients.index)

client = clients.loc[selected_id]
st.subheader("Informations client")
st.write(client)

if st.button("📤 Envoyer pour scoring"):
    input_data = {
        "age": int(client["age"]),
        "revenu": float(client["revenu"]),
        "anciennete": int(client["anciennete"]),
        "nb_incidents": int(client["nb_incidents"]),
        "score_credit": float(client["score_credit"]),
    }

    try:
        res = requests.post("https://neo-banque-dashboard.onrender.com", json=input_data)
        response_data = res.json()
        score = response_data["score"]
        explanations_from_api = response_data.get("explanations", [])  # clé au pluriel

        st.metric("Score d’éligibilité au prêt", f"{score * 100:.1f} %")

        if score > 0.5:
            st.success("✅ Client éligible probable au prêt.")
        else:
            st.warning("⚠️ Client potentiellement inéligible ou profil à risque.")

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
        st.error("Erreur de connexion à l’API. Assurez-vous que l'API est lancée sur http://localhost:8000.")
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite : {e}")
