#### English Version

# Streamlit Web Application for Predicting ONCF Trains Occupancy Levels

## Overview

This repository contains a Streamlit web application designed to predict the occupancy levels of ONCF trains. The application leverages machine learning to forecast train occupancy, helping travelers choose less crowded trains and therefore optimizing the overall rail traffic.

## Setup Guide

To run this application locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Chenguiti6444/ONCF-Affluence.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd ONCF-Affluence
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:
    ```bash
    streamlit run App.py
    ```

## Usage

- Launch the application using the `streamlit run App.py` command.
- Use the interface to select travel parameters and view occupancy predictions.

## File Descriptions

- **App.py**: Main script to run the Streamlit app.
- **model_catboost_4.zip**: CatBoost model trained on data from the ONCF PIS (Passenger Information System) for train occupancy prediction.
- **requirements.txt**: A text file listing all the Python libraries and dependencies required to run the application. Installing these ensures the environment is correctly set up for the Streamlit app.
- **LICENSE**: The license file for the project, indicating the legal terms under which the project can be used, modified, and distributed. This project uses the MIT License.
- **.gitignore**: A file specifying which files and directories should be ignored by Git version control.
- **.devcontainer/**: Directory containing configuration files for developing inside a container. This is useful for ensuring a consistent development environment across different machines.
- **Images/**: A folder containing images used in the project (e.g., logos, icons).
- **Aggregation_des_N°_de_Trains_par_OD+Gamme.csv**: A CSV file that associates each triplet of origin-destination (OD), train type (Gamme), and departure time (Heure) with the available train numbers for that triplet. It is used to filter the dropdown list displayed in the ONCF Affluence interface, allowing users to select valid train numbers.


## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## Contact

For questions or suggestions, please contact chenguiti.elmehdi@gmail.com

---

#### Version Française

# Application Web Streamlit de Prédiction d'Affluence à Bord des Trains ONCF

## Introduction

Ce dépôt contient une application web Streamlit conçue pour prédire le niveau d'affluence à bord des trains ONCF. L'application utilise l'apprentissage automatique pour prévoir l'affluence à bord des trains, aidant ainsi les voyageurs à choisir des trains moins bondés tout en optimisant le trafic ferroviaire.

## Guide d'installation

Pour exécuter cette application localement, suivez ces étapes :

1. **Clonez le dépôt** :
    ```bash
    git clone https://github.com/Chenguiti6444/ONCF-Affluence.git
    ```

2. **Accédez au répertoire du projet** :
    ```bash
    cd ONCF-Affluence
    ```

3. **Installez les packages requis** :
    ```bash
    pip install -r requirements.txt
    ```

4. **Exécutez l'application** :
    ```bash
    streamlit run App.py
    ```

## Utilisation

- Lancez l'application en utilisant la commande `streamlit run App.py`.
- Utilisez l'interface pour sélectionner les paramètres de voyage et visualiser les prédictions d'affluence.

## Description des Fichiers

- **App.py** : Script principal pour exécuter l'application Streamlit.
- **model_catboost_4.zip** : Modèle CatBoost entraîné sur des données issues du Système d'Information Voyageurs (SIV) de l'ONCF pour prédire l'affluence à bord des trains.
- **requirements.txt** : Un fichier texte listant toutes les bibliothèques Python et dépendances nécessaires pour exécuter l'application. L'installation de ces dépendances garantit que l'environnement est correctement configuré pour l'application Streamlit.
- **LICENSE** : Le fichier de licence du projet, indiquant les conditions légales sous lesquelles le projet peut être utilisé, modifié et distribué. Ce projet utilise la licence MIT.
- **.gitignore** : Un fichier spécifiant quels fichiers et répertoires doivent être ignorés par le contrôle de version Git.
- **.devcontainer/** : Répertoire contenant les fichiers de configuration pour le développement dans un conteneur. Ceci est utile pour assurer un environnement de développement cohérent sur différentes machines.
- **Images/** : Un dossier contenant les images utilisées dans le projet (par exemple, logos, icônes).
- **Aggregation_des_N°_de_Trains_par_OD+Gamme.csv** : Un fichier CSV qui associe chaque triplet composé d'une paire origine-destination (OD), d'une gamme de train (Gamme), et d'une heure de départ (Heure) avec le ou les numéros de trains disponibles pour ce triplet. Il est utilisé pour filtrer la liste déroulante affichée au niveau de l'interface de l'application ONCF Affluence, permettant aux utilisateurs de sélectionner des numéros de trains valides.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## Contribution

Les contributions sont les bienvenues ! Veuillez soumettre une pull request ou ouvrir un problème pour discuter des modifications.

## Contact

Pour des questions ou suggestions, veuillez contacter chenguiti.elmehdi@gmail.com