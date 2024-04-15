import streamlit as st
import pandas as pd
from datetime import datetime
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# Chargez les données
data = pd.read_csv('Aggregation des N° de Trains par OD+Gamme+Heure.csv')

def update_heures(od_choice):
    # Filtrez les données pour l'OD sélectionné
    filtered_data = data[data['OD'] == od_choice]
    # Extrayez les valeurs uniques des heures, triez-les et convertissez-les en chaînes de caractères
    heures = sorted(filtered_data['Heure'].dropna().unique().astype(str).tolist())
    return heures

def update_gammes(od_choice, heure_choice):
    # Filtrez les données pour l'OD et l'heure sélectionnés
    filtered_data = data[(data['OD'] == od_choice) & (data['Heure'] == int(heure_choice))]
    # Extrayez les valeurs uniques de gamme, triez-les et convertissez-les en chaînes de caractères
    gammes = sorted(filtered_data['Gamme'].dropna().unique().astype(str).tolist())
    return gammes

def update_num_train(od_choice, heure_choice, gamme_choice):
    # Filtrez les données pour l'OD, l'heure et la gamme sélectionnés
    filtered_data = data[(data['OD'] == od_choice) & 
                         (data['Heure'] == int(heure_choice)) & 
                         (data['Gamme'] == gamme_choice)]
    # Extrayez les numéros de train uniques, séparez-les par virgule, aplatissez la liste et triez
    num_trains = sorted(set([train.strip() for sublist in filtered_data['N° de train'].dropna().str.split(',') for train in sublist]))
    return num_trains

# Créez l'interface Streamlit
st.title("Prédicteur d'Affluence Train")

# Ajoutez un widget de sélection de date
selected_date = st.date_input("Date", min_value=datetime.today())

# Ajoutez une boîte de sélection pour OD
od_choice = st.selectbox("OD", sorted(data['OD'].unique().tolist()), index=0)

# Basé sur le choix de l'OD, mettez à jour et ajoutez une boîte de sélection pour Heure
heures = update_heures(od_choice)
heure_choice = st.selectbox("Heure", heures, index=0 if heures else None)

# Basé sur le choix de l'OD et l'Heure, mettez à jour et ajoutez une boîte de sélection pour Gamme
gammes = update_gammes(od_choice, heure_choice)
gamme_choice = st.selectbox("Gamme", gammes, index=0 if gammes else None)

# Basé sur le choix de l'OD, l'Heure et la Gamme, mettez à jour et ajoutez une boîte de sélection pour Numéro de train
num_trains = update_num_train(od_choice, heure_choice, gamme_choice)
num_train_choice = st.selectbox("Numéro de train", num_trains, index=0 if num_trains else None)

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

# Initialisation des préprocesseurs et du modèle
model_path = 'RegDNN_model.tf'
label_encoders_path = 'label_encoders.pkl'
scaler_path = 'RegDNN_features_scaler.pkl'
model, label_encoders, scaler = load_model_and_preprocessors(model_path, label_encoders_path, scaler_path)

# Bouton de prédiction
if st.button('Prédire l\'affluence'):
    # Prétraitement des entrées
    inputs_preprocessed = preprocess_inputs(heure_choice, gamme_choice, num_train_choice, selected_date, label_encoders, scaler)
    
    # Faire la prédiction
    prediction = model.predict(inputs_preprocessed)
    prediction_value=prediction[0]
    
    if prediction_value < 0.10:
        st.image("Images/Affluence Faible.png")  # Adjust path as needed
    elif prediction_value < 0.20:
        st.image("Images/Affluence Moyenne.png")  # Adjust path as needed
    else:
        st.image("Images/Affluence Forte.png")  # Adjust path as needed