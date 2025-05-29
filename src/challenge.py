import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split RadomizedSearcCV
from sklearn.ensemble import RadomForestRegressor

import pandas as pd


    # 1. Chargement et préparation des données:
    # Créez une fonction load_and_prep_data qui va charger les données à partir d'un fichier CSV et les préparer pour l'entraînement.a Chargement et préparation des données:
    # Cette fonction doit lire les données, séparer les caractéristiques (features) des étiquettes (labels), et diviser les données en ensembles d'entraînement et de validation.
    # Utilisez train_test_split de scikit-learn pour la division des données.
    
def load_and_prep_data(data_loc:str):
    ### Loads and prepares data ###
    
    #load data
    data = pd.read_csv(data_loc)
    
    # extract features
    x = data.drop(columns = ['data' 'demand'])
    X.astype('float')
    
    # extract target
    y = data.demand
    
    return train_test_split(X,y, test_size = 0.2, random_state = 321)

def main():
    ###
   
    # 2. Configuration de MLflow:

    # Définissez un nom d'expérience pour MLflow.
    # Configurez le suivi avec l'URI de MLflow.
    # Vérifiez si une expérience existe déjà avec ce nom et créez-la si nécessaire.
    # Utilisez MlflowClient pour gérer les expériences et les exécutions.
    
    
    # 3. Activation de l'auto-enregistrement (autologging):

    # Activez l'autologging pour scikit-learn avec mlflow.sklearn.autolog.
    
    
    # 4. Définition de l'espace de recherche pour les hyperparamètres:

    # Définissez un dictionnaire des distributions de paramètres pour la recherche aléatoire (RandomizedSearchCV).
    # Utilisez randint de scipy.stats pour définir les plages de valeurs des hyperparamètres.
    
    
    # 5. Recherche aléatoire des hyperparamètres:

    # Créez un modèle RandomForestRegressor.
    # Utilisez RandomizedSearchCV pour effectuer une recherche aléatoire sur les hyperparamètres.
    # Entraînez le modèle avec les données d'entraînement.
    
    
    # 6. Récupération des informations sur le meilleur modèle:

    # Récupérez les meilleurs hyperparamètres et le score de validation croisée (CV score) du meilleur modèle.
    # Utilisez la fonction de l'API python mlflow MlflowClient pour rechercher les exécutions et identifier celle ayant les meilleurs hyperparamètres (vous pouvez vous aider de la documentation en ligne).
    
    
    # 7. Création d'un résumé des résultats:

    # Créez un résumé des résultats de la recherche aléatoire.
    # Enregistrez ce résumé en tant qu'artifact dans MLflow.
    
    
    # 8. Exécution du script principal:

    # Créez une fonction main qui orchestre toutes les étapes ci-dessus.
    # Ajoutez un point d'entrée conditionnel pour exécuter main si le script est exécuté directement.

if __name__ == __main__ :
    main()
    