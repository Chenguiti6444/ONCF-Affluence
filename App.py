import streamlit as st
import pandas as pd
import itertools
from datetime import datetime
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Chargez les données
data = pd.read_csv('Aggregation des N° de Trains par OD+Gamme+Heure.csv')

# Chargement du modèle et des préprocesseurs
@st.cache_resource
def load_model_and_preprocessors(model_path, label_encoders_path, scaler_path):
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = tf.keras.models.load_model(model_path, options=load_options)
    with open(label_encoders_path, 'rb') as le_file:
        label_encoders = pickle.load(le_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, label_encoders, scaler


# Prétraitement des entrées pour le modèle
def preprocess_inputs(heure_choice, gamme_choice, num_train_choice, selected_date, label_encoders, scaler):
    # Convertir les entrées en format attendu par le modèle
    inputs = pd.DataFrame({
        'Nº de train': [num_train_choice],
        'Gamme': [gamme_choice],
        'Heure': [int(heure_choice)],
        'Jour_Semaine': [selected_date.weekday()],
        'Mois': [selected_date.month]
    })
    
    # Appliquer l'encodage des labels
    for col in ['Gamme', 'Nº de train']:
        inputs[col] = label_encoders[col].transform(inputs[col])
    
    # Appliquer la mise à l'échelle des données numériques
    inputs = scaler.transform(inputs)
    return inputs


def update_heures(od_choice, selected_date):
    # Filtrez les données pour l'OD sélectionné
    filtered_data = data[data['OD'] == od_choice]
    # Extrayez les valeurs uniques des heures, triez-les
    heures = sorted(filtered_data['Heure'].dropna().unique().astype(int).tolist())
    
    # Filtrer les heures pour ne garder que celles après l'heure courante si la date sélectionnée est aujourd'hui
    if selected_date == datetime.today().date():
        current_hour = datetime.now().hour
        heures = [heure for heure in heures if heure >= current_hour]
        
    # Convertissez-les en chaînes de caractères pour l'affichage
    heures = [str(heure) for heure in heures]
    return heures


def update_gammes(od_choice, heure_choice):
    if heure_choice is None:
        return []  # Return an empty list if heure_choice is None
    # Filtrez les données pour l'OD et l'heure sélectionnés
    filtered_data = data[(data['OD'] == od_choice) & (data['Heure'] == int(heure_choice))]
    # Extrayez les valeurs uniques de gamme, triez-les et convertissez-les en chaînes de caractères
    gammes = sorted(filtered_data['Gamme'].dropna().unique().astype(str).tolist())
    return gammes

def update_num_train(od_choice, heure_choice, gamme_choice):
    if heure_choice is None or gamme_choice is None:
        return []  # Return an empty list if heure_choice or gamme_choice is None
    # Filtrez les données pour l'OD, l'heure et la gamme sélectionnés
    filtered_data = data[(data['OD'] == od_choice) & 
                         (data['Heure'] == int(heure_choice)) & 
                         (data['Gamme'] == gamme_choice)]
    # Extrayez les numéros de train uniques, séparez-les par virgule, aplatissez la liste et triez
    num_trains = sorted(set([train.strip() for sublist in filtered_data['N° de train'].dropna().str.split(',') for train in sublist]))
    return num_trains


st.set_page_config(layout="wide")

# Chargement du logo et affichage en entête de page
logo = Image.open('Images/Logo.png')
st.image(logo, width=100)

# Utilisation de st.columns pour diviser la page en deux colonnes
col1, col2 = st.columns(2)

with col1:
    # Sélection de la date
    selected_date = st.date_input("Date de voyage", min_value=datetime.today(), value=None,help="Veuillez choisir une date")

    # Si une date est choisie, affichez la sélection d'OD
    if selected_date:
        od_choice = st.selectbox("Origine-Destination", sorted(data['OD'].unique().tolist()), index=None,placeholder="Veuillez choisir l'OD")
    else:
        od_choice = None

    # Si un OD est choisi, affichez la sélection d'heure
    if od_choice:
        heures = update_heures(od_choice, selected_date)
        heure_choice = st.selectbox("Heure de voyage", heures, index=None,placeholder="Veuillez choisir l'heure")
    else:
        heure_choice = None

with col2:
    # Si une heure est choisie, affichez la sélection de gamme
    if heure_choice:
        gammes = update_gammes(od_choice, heure_choice)
        gamme_choice = st.selectbox("Gamme", gammes, index=None,placeholder="Veuillez choisir la gamme")
    else:
        gamme_choice = None

    # Si une gamme est choisie, affichez la sélection du numéro de train
    if gamme_choice:
        num_trains = update_num_train(od_choice, heure_choice, gamme_choice)
        num_train_choice = st.selectbox("Numéro de train", num_trains, index=0)
    else:
        num_train_choice = None

# Initialisation des préprocesseurs et du modèle
model_path = 'RegDNN_model.tf'
label_encoders_path = 'label_encoders.pkl'
scaler_path = 'RegDNN_features_scaler.pkl'
model, label_encoders, scaler = load_model_and_preprocessors(model_path, label_encoders_path, scaler_path)

# Calculer toutes les prédictions possibles
def calculate_all_predictions(date, od_choice,heure_choice):
    # Convertir l'heure choisie en integer pour la comparaison
    selected_heure_int = int(heure_choice)
    heures = [heure for heure in update_heures(od_choice, date) if int(heure) > selected_heure_int]
    all_combinations = []
    results = []

    # Obtenir toutes les gammes et numéros de train pour chaque heure
    for heure in heures:
        gammes = update_gammes(od_choice, heure)
        for gamme in gammes:
            num_trains = update_num_train(od_choice, heure, gamme)
            # Créer des combinaisons de heure, gamme et numéro de train
            combinations = list(itertools.product([heure], [gamme], num_trains))
            all_combinations.extend(combinations)

    # Faire des prédictions pour chaque combinaison
    for heure, gamme, num_train in all_combinations:
        # Préparation des entrées pour le modèle
        inputs_preprocessed = preprocess_inputs(heure, gamme, num_train, date, label_encoders, scaler)
        prediction = model.predict(inputs_preprocessed)[0]

        # Stocker les résultats avec leurs caractéristiques
        results.append({'Heure': heure, 'Gamme': gamme, 'Numéro de Train': num_train, 'Prédiction': prediction})

    return results

# Afficher les résultats
def display_results(results):
    # Convertir en DataFrame
    results_df = pd.DataFrame(results)
    # Trier par prédiction pour obtenir les affluences les plus faibles
    sorted_results = results_df.sort_values(by='Prédiction').head(5)
    st.table(sorted_results)

# Bouton de prédiction, actif seulement si tous les champs sont remplis
with col2:
    if selected_date and od_choice and heure_choice and gamme_choice and num_train_choice:
        if st.button('Prédire l\'affluence'):
            inputs_preprocessed = preprocess_inputs(heure_choice, gamme_choice, num_train_choice, selected_date, label_encoders, scaler)
            prediction = model.predict(inputs_preprocessed)
            prediction_value = prediction[0]

            if prediction_value < 0.10:
                st.image("Images/Affluence Faible.png", width=200)
            elif prediction_value < 0.22:
                st.image("Images/Affluence Moyenne.png", width=200)
            else:
                st.image("Images/Affluence Forte.png", width=200)
            if selected_date and od_choice and num_train_choice and prediction_value>0.10:
                results = calculate_all_predictions(selected_date, od_choice,heure_choice)

                # Ensuite, comparez les prédictions et calculez le pourcentage de différence
                results_with_comparison = []

                for result in results:
                    # Extraire le scalaire de la prédiction avant de faire le calcul
                    prediction_result = result['Prédiction']
                    difference_percentage = ((prediction_value - prediction_result) / prediction_value) * 100
                    # Ici, nous convertissons le pourcentage en scalaire pour éviter les erreurs de formatage
                    difference_percentage_value = round(difference_percentage.item())
                    result["Différence d'Affluence"] = f"{difference_percentage_value}% moins chargé que le train choisi"
                    results_with_comparison.append(result)

                # Triez les résultats pour obtenir les prédictions les plus faibles par rapport à la référence
                results_df = pd.DataFrame(results_with_comparison)
                sorted_results = results_df.sort_values(by='Prédiction').head(5)
                # Réinitialiser l'index du DataFrame et supprimer l'ancien index
                sorted_results = sorted_results.reset_index(drop=True)

                with col1:
                    st.subheader('Top 5 des Trains les Moins Chargés')
                    # Affichez les résultats dans un tableau
                    st.dataframe(sorted_results[['Heure', 'Gamme', 'Numéro de Train', "Différence d'Affluence"]],hide_index=True)


