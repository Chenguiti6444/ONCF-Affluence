import streamlit as st
import pandas as pd
import itertools
import zipfile
from datetime import datetime
import pickle
from catboost import CatBoostRegressor
import numpy as np
from PIL import Image

# Chemin du logo ONCF-Affluence
logo = Image.open('Images/Logo.png')
favicon=Image.open('Images/Favicon.png')

# Configuration de la page
st.set_page_config(page_title="ONCF-Affluence", page_icon=favicon, layout="wide")

# Chargez les données
data = pd.read_csv('Aggregation_des_N°_de_Trains_par_OD+Gamme+Heure_New.csv')

# Séparation des colonnes 'Origine' et 'Destination'
data[['Origine', 'Destination']] = data['OD'].str.split(' - ', expand=True)
origines = sorted([origine.strip() for origine in data['Origine'].unique()])
destinations = sorted([destination.strip() for destination in data['Destination'].unique()])


# Fonction pour obtenir la saison
def get_season(date):
    year = date.year
    seasons = {
        "winter_start": datetime(year, 12, 21).date(),
        "spring_start": datetime(year, 3, 20).date(),
        "summer_start": datetime(year, 6, 21).date(),
        "fall_start": datetime(year, 9, 22).date()
    }
    
    if date >= seasons["winter_start"] or date < seasons["spring_start"]:
        return 0  # Hiver
    elif date >= seasons["spring_start"] and date < seasons["summer_start"]:
        return 1  # Printemps
    elif date >= seasons["summer_start"] and date < seasons["fall_start"]:
        return 2  # Été
    else:
        return 3  # Automne

    
# Chargement du modèle
@st.cache_resource
def load_model(model_zip_path):
    # Extract the model file from the zip archive
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    
    # Load the model from the extracted file
    model = CatBoostRegressor()
    model.load_model("model_catboost_4.cbm")  # Assuming the model file is named model_catboost_4.cbm
    
    return model


# Prétraitement des entrées pour le modèle
def preprocess_inputs(heure_choice, gamme_choice, num_train_choice, selected_date):
    semaine = selected_date.isocalendar().week
    année = selected_date.isocalendar().year
    jour_semaine = selected_date.isocalendar().weekday
    saison = get_season(selected_date)
    
    inputs = pd.DataFrame({
        'Nº de train': [num_train_choice],
        'Gamme': [gamme_choice],
        'Heure': [int(heure_choice)],
        'Semaine': [semaine],
        'Jour_Semaine': [jour_semaine],
        'Année': [année],
        'Saison': [saison]
    })
    
    return inputs


# Fonction pour mettre à jour les destinations basées sur l'origine sélectionnée
def updateDestination(origine):
    filtered_data = data[data['Origine'] == origine]
    unique_destinations = sorted(filtered_data['Destination'].unique())
    return unique_destinations

# Fonction pour mettre à jour les heures basées sur l'Origine et la Destination
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

# Fonction pour mettre à jour les gammes basées sur l'Origine, la Destination et l'Heure
def update_gammes(od_choice, heure_choice):
    # Filtrez les données pour l'OD et l'heure sélectionnés
    filtered_data = data[(data['OD'] == od_choice) & (data['Heure'] == int(heure_choice))]
    # Extrayez les valeurs uniques de gamme, triez-les et convertissez-les en chaînes de caractères
    gammes = sorted(filtered_data['Gamme'].dropna().unique().astype(str).tolist())
    return gammes

# Fonction pour mettre à jour les numéros de train basés sur l'Origine, la Destination, l'Heure et la Gamme
def update_num_train(od_choice, heure_choice, gamme_choice):
    # Filtrez les données pour l'OD, l'heure et la gamme sélectionnés
    filtered_data = data[(data['OD'] == od_choice) & 
                         (data['Heure'] == int(heure_choice)) & 
                         (data['Gamme'] == gamme_choice)]
    # Extrayez les numéros de train uniques, séparez-les par virgule, aplatissez la liste et triez
    num_trains = sorted(set([train.strip() for sublist in filtered_data['N° de train'].dropna().str.split(',') for train in sublist]))
    return num_trains

# Initialisation du modèle
model_zip_path = 'model_catboost_4.zip'
model = load_model(model_zip_path)

# Calculer toutes les prédictions possibles
def calculate_all_predictions(date, od_choice, heure_choice, model):
    # Convertir l'heure choisie en integer pour la comparaison
    selected_heure_int = int(heure_choice)
    # Définir l'intervalle de 3 heures avant et après l'heure choisie
    interval_start = selected_heure_int - 3
    interval_end = selected_heure_int + 3
    # Obtenir toutes les heures possibles pour la destination choisie
    all_heures = update_heures(od_choice, heure_choice)

    # Si le nombre d'heures disponibles est >= 3, appliquer le filtrage avec intervalle
    if len(all_heures) >= 4:
        # Filtrer les heures dans l'intervalle spécifié
        heures = [heure for heure in all_heures if (interval_start <= int(heure) <= interval_end)]
    else:
        # Si moins de 4 heures disponibles, ne pas filtrer par intervalle
        heures = [heure for heure in all_heures]
    all_combinations = []  # Stocker toutes les combinaisons possibles d'heure, gamme et numéro de train
    results = []  # Stocker les résultats des prédictions

    # Pour chaque heure possible
    for heure in heures:
        # Obtenir toutes les gammes pour cette heure
        gammes = update_gammes(od_choice, heure)
        # Filtrer les gammes pour exclure "AL BORAQ" si la gamme choisie n'est pas "AL BORAQ"
        if gamme_choice != "AL BORAQ":
            gammes = [gamme for gamme in gammes if gamme != "AL BORAQ"]
        if gamme_choice == "AL BORAQ":
            gammes = [gamme for gamme in gammes if gamme == "AL BORAQ"]
        # Pour chaque gamme
        for gamme in gammes:
            # Obtenir tous les numéros de train pour cette heure et cette gamme
            num_trains = update_num_train(od_choice, heure, gamme)
            # Créer toutes les combinaisons possibles d'heure, gamme et numéro de train
            combinations = list(itertools.product([heure], [gamme], num_trains))
            all_combinations.extend(combinations)  # Ajouter les combinaisons à la liste totale

    # Pour chaque combinaison possible
    for heure, gamme, num_train in all_combinations:
        # Prétraiter les entrées pour les rendre compatibles avec le modèle
        inputs_preprocessed = preprocess_inputs(heure, gamme, num_train, date)
        # Faire une prédiction avec le modèle
        prediction = model.predict(inputs_preprocessed)[0]
        # Stocker les résultats avec les caractéristiques correspondantes
        results.append({'Heure': heure, 'Gamme': gamme, 'N° de Train': num_train, 'Prédiction': prediction})

    return results

# Afficher les résultats
def display_results(results):
    # Convertir en DataFrame
    results_df = pd.DataFrame(results)
    # Trier par prédiction pour obtenir les affluences les plus faibles
    sorted_results = results_df.sort_values(by='Prédiction').head(5)
    st.table(sorted_results)


st.image(logo, width=200)

# Utilisation de st.columns pour diviser la page en deux colonnes
col1, col2 = st.columns(2)

# Sélections de date, heure, gamme, et numéro de train
with col1:  
    selected_date = st.date_input("Date de voyage", min_value=datetime.today(),value=None,help="Veuillez choisir une date")
    origines = sorted(data['Origine'].unique())
    col3, col4 = st.columns(2)
    with col3:
        if selected_date: 
            origine_choice = st.selectbox("Gare de départ", origines, index=None,placeholder="Veuillez choisir une gare")
            if origine_choice:
                with col4:
                    destination_choice = st.selectbox("Gare d'arrivée", updateDestination(origine_choice), index=None,placeholder="Veuillez choisir une gare")
                    if destination_choice:
                        od_choice = f"{origine_choice} - {destination_choice}"
                        with col2:
                            # Utilisez la fonction update_heures pour obtenir la liste des heures
                            heures_list = update_heures(od_choice, selected_date)
                            # Vérifiez si la liste des heures est vide avant d'afficher le selectbox
                            if heures_list:
                                heure_choice = st.selectbox("Heure de voyage", heures_list, index=None, placeholder="Veuillez choisir l'heure")
                                if heure_choice:
                                    gamme_choice = st.selectbox("Gamme", update_gammes(od_choice, heure_choice), index=None, placeholder="Veuillez choisir la gamme")
                                    if gamme_choice:
                                        num_train_choice = st.selectbox("Numéro de train", update_num_train(od_choice, heure_choice, gamme_choice), index=0)
                                        # Bouton de prédiction, actif seulement si tous les champs sont remplis

                                        if selected_date and od_choice and heure_choice and gamme_choice and num_train_choice:
                                            if st.button('Prédire l\'affluence'):
                                                inputs_preprocessed = preprocess_inputs(heure_choice, gamme_choice, num_train_choice, selected_date)
                                                prediction = model.predict(inputs_preprocessed)
                                                prediction_value = prediction[0]
                                                #print(prediction_value)
                                                if prediction_value < 0.105:
                                                    st.image("Images/Affluence Faible.png", width=200)
                                                elif prediction_value <= 0.46:
                                                    st.image("Images/Affluence Moyenne.png", width=200)
                                                else:
                                                    st.image("Images/Affluence Forte.png", width=200)
                                                if selected_date and od_choice and num_train_choice and prediction_value>=0.105:
                                                    results = calculate_all_predictions(selected_date, od_choice,heure_choice,model)

                                                    # Ensuite, comparez les prédictions et calculez le pourcentage de différence
                                                    results_with_comparison = []
                                                    for result in results:
                                                        # Limiter la valeur minimale de prediction_result
                                                        result['Prédiction'] = max(result['Prédiction'], 0.003756)

                                                        prediction_result = result['Prédiction']
                                                        difference_percentage = ((prediction_value - prediction_result) / prediction_value) * 100
                                                        # Ici, on convertit le pourcentage en scalaire pour éviter les erreurs de formatage et on l'arrondit
                                                        difference_percentage_value = round(difference_percentage, 1)
                                                        if difference_percentage_value>0:
                                                            result["Différence d'Affluence"] = f"{difference_percentage_value}% moins chargé que le train choisi"
                                                            result["Pourcentage"]=difference_percentage_value
                                                            results_with_comparison.append(result)
                                                
                                                    results_df = pd.DataFrame(results_with_comparison)
                                                    
                                                    if not results_df.empty:
                                                        # Trier les résultats par pourcentage décroissant puis par Heure (croissant)
                                                        # Convertir la colonne Heure en entier
                                                        results_df['Heure'] = results_df['Heure'].astype(int)
                                                        sorted_results = results_df.sort_values(['Pourcentage'], ascending=False).head(5)
                                                        sorted_results.sort_values(['Heure'], ascending=True,inplace=True)

                                                        with col1:
                                                            st.subheader('Suggestions de Trains Moins Chargés')
                                                            # Affichez les résultats dans un tableau
                                                            st.dataframe(sorted_results[['Heure', 'Gamme', 'N° de Train', "Différence d'Affluence"]],hide_index=True)
                            else:
                                st.warning("Aucun train n'est disponible aujourd'hui pour le trajet sélectionné.")

