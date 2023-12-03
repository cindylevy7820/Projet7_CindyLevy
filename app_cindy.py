import pandas as pd
import streamlit as st
import requests
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import json
import shap
import lightgbm as lgb

def plot_indicator_comparison(data, client_choice, indicator):
    # Obtenir l'indicateur pour le client sélectionné
    client_indicator = data.loc[client_choice, indicator]

    # Calculer la moyenne de l'indicateur pour les autres clients en cours
    other_clients_mean = np.mean(data[data.index != client_choice][indicator])

    # Créer le graphique à barres comparatif
    fig, ax = plt.subplots()
    ax.bar(['Client sélectionné', 'Moyenne des autres clients'], [client_indicator, other_clients_mean])
    ax.set_ylabel(indicator)
    ax.set_title('Comparaison de l\'indicateur')
    st.pyplot(fig)

###### LES FONCTIONS
@st.cache_data
def datas():
    data = pd.read_csv('data/default_risk.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    sample = pd.read_csv('data/X_sample.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    return data, sample

data, sample = datas()

def models():
    pickle_in = open('model/model.pkl', 'rb') 
    prediction = pickle.load(pickle_in)
    return prediction

model = models()

def interpret_score(score):
    if score >= 0.7:
        color = "red"
        message = "Le client présente un risque élevé de faillite."
    elif score >= 0.4:
        color = "orange"
        message = "Le client présente un risque modéré de faillite."
    else:
        color = "green"
        message = "Le client présente un faible risque de faillite."
    
    return f"<span style='color:{color}'>{message}</span>"

def load_predictions_API(sample, id, clf):
    X=sample.iloc[:, :-1]
    score = clf.predict_proba(X[X.index == int(id)])[:,1]
    return score

###### CODE PRINCIPALE
def main():
    
    ### BAR MENU
    client_choice = st.sidebar.selectbox(
        'Veuillez choisir un client', sample.index.values)
        
     ### CONTENU DU DASHBOARD
    header = "<h1 align='center'>SCORING CREDIT</h1>"
    st.write(header, unsafe_allow_html=True)
    
    # Probabilité remboursement et interprétation
    prediction = load_predictions_API(sample, client_choice, model)
    st.write("<B><h5>Probabilité de faillite du client {} est de : {:.0f} % </B></h5>".format(client_choice, round(float(prediction)*100, 2)), unsafe_allow_html=True)
    st.write("<p>{}</p>".format(interpret_score(prediction)), unsafe_allow_html=True)
    
    # Informations clients
    client_info = data.loc[client_choice]
    st.write("<h3><B>Informations principales du client :</B></h3>", unsafe_allow_html=True)
    st.write(f"<b>- GENRE :</b> {client_info['CODE_GENDER']} ", unsafe_allow_html=True)
    st.write(f"<b>- POSSEDE UNE VOITURE :</b> {client_info['FLAG_OWN_CAR']} ", unsafe_allow_html=True)
    st.write(f"<b>- POSSEDE UN BIEN IMMOBILIER :</b> {client_info['FLAG_OWN_REALTY']} ", unsafe_allow_html=True)
    st.write(f"<b>- AGE :</b> {int(client_info['DAYS_BIRTH']/365)} ans", unsafe_allow_html=True)
    st.write(f"<b>- REVENU TOTAL :</b> {client_info['AMT_INCOME_TOTAL']} ", unsafe_allow_html=True)
    st.write(f"<b>- SITUATION FAMILIAL :</b> {client_info['NAME_FAMILY_STATUS']} ", unsafe_allow_html=True)
    st.write(f"<b>- NOMBRE ENFANT :</b> {client_info['CNT_CHILDREN']} ", unsafe_allow_html=True)
    st.write(f"<b>- POURCENTAGE MONTANT CREDIT (/revenu client) :</b> {client_info['CREDIT_INCOME_PERCENT']} ", unsafe_allow_html=True)
   
    # Informations sur tous les clients
    filtered_data = data.copy()  # Copie des données pour les filtrer
    # Options de filtrage
    filter_options = st.multiselect('Variables de filtrage', data.columns)
    # Appliquer les filtres
    for option in filter_options:
        filter_value = st.selectbox(f"Valeur de {option}", data[option].unique())
        filtered_data = filtered_data[filtered_data[option] == filter_value]
    # Affichage des informations descriptives filtrées
    st.subheader('Informations descriptives filtrées')
    st.dataframe(filtered_data)

    # Afficher les graphiques
    #Age distribution plot
    data_AMT_INCOME_TOTAL = data["AMT_INCOME_TOTAL"]
    fig, ax = plt.subplots()
    sns.histplot(data_AMT_INCOME_TOTAL, bins=100)
    ax.axvline(client_info["AMT_INCOME_TOTAL"], color="green", linestyle='--')
    ax.set(title='Revenu', xlabel='Revenu', ylabel='Nombre')
    st.pyplot(fig)   
    st.subheader('Comparaison de l\'indicateur')
    indicator_choice = st.selectbox('Sélectionnez un indicateur', ['AMT_INCOME_TOTAL', 'CNT_CHILDREN'])
    plot_indicator_comparison(data, client_choice, indicator_choice)
    
    # Analyse bi-dimmentionnel (Graphique revenu en fonction du nombre d'enfants)
    st.subheader('Analyse bi-dimmentionnel')
    fig, ax = plt.subplots()
    sns.scatterplot(x=data["CNT_CHILDREN"], y=data["AMT_INCOME_TOTAL"])
    ax.set(title='Revenu en fonction du nombre d\'enfants', xlabel='Nombre d\'enfants', ylabel='Revenu')
    st.pyplot(fig)

    # SHAP
    shap.initjs()
    X = sample.iloc[:, :-1]
    X = X[X.index == client_choice]
    fig, ax = plt.subplots(figsize=(10, 10))
    explainer = shap.TreeExplainer(models())
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, max_display=7, plot_size=(5, 5))
    st.pyplot(fig)

if __name__ == '__main__':
    main()
