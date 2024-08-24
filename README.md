#### English Version

# Streamlit Web Application for Predicting ONCF Trains Occupancy Levels

## Overview

This repository contains a Streamlit web application designed to predict the occupancy levels of ONCF trains. The application leverages machine learning to forecast train occupancy, helping travelers choose less crowded trains and therefore optimizing the overall rail traffic.

## Setup Guide

To run this application locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Chenguiti6444/ONCF-Trains-Occupancy-Prediction.git
    cd ONCF-Trains-Occupancy-Prediction
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    streamlit run App.py
    ```

## Usage

- Launch the application using the `streamlit run App.py` command.
- Use the interface to select travel parameters and view occupancy predictions.

## File Descriptions

- **App.py**: Main script to run the Streamlit app.
- **model_catboost_4.zip**: Pre-trained CatBoost model for occupancy prediction.
- **encoder.pkl**: Encoder used for transforming categorical data.
- **requirements.txt**: List of dependencies required to run the application.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## Contact

For questions or suggestions, please contact [chenguiti.elmehdi@gmail.com].

---

#### Version Française

# Application Web Streamlit de Prédiction d'Affluence à Bord des Trains ONCF

## Introduction

Ce dépôt contient une application web Streamlit conçue pour prédire le niveau d'affluence à bord des trains ONCF. L'application utilise l'apprentissage automatique pour prévoir l'affluence à bord des trains, aidant ainsi les voyageurs à choisir des trains moins bondés tout en optimisant le trafic ferroviaire.

## Guide d'installation

Pour exécuter cette application localement, suivez ces étapes :

1. **Clonez le dépôt** :
    ```bash
    git clone https://github.com/Chenguiti6444/ONCF-Trains-Occupancy-Prediction.git
    cd ONCF-Trains-Occupancy-Prediction
    ```

2. **Installez les packages requis** :
    ```bash
    pip install -r requirements.txt
    ```

3. **Exécutez l'application** :
    ```bash
    streamlit run App.py
    ```

## Utilisation

- Lancez l'application en utilisant la commande `streamlit run App.py`.
- Utilisez l'interface pour sélectionner les paramètres de voyage et visualiser les prédictions d'affluence.

## Description des Fichiers

- **App.py** : Script principal pour exécuter l'application Streamlit.
- **model_catboost_4.zip** : Modèle CatBoost pré-entraîné pour la prédiction d'affluence.
- **encoder.pkl** : Encodeur utilisé pour transformer les données catégorielles.
- **requirements.txt** : Liste des dépendances nécessaires pour exécuter l'application.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contribution

Les contributions sont les bienvenues ! Veuillez soumettre une pull request ou ouvrir un problème pour discuter des modifications.

## Contact

Pour des questions ou suggestions, veuillez contacter [chenguiti.elmehdi@gmail.com].
