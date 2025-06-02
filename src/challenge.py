import mlflow
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

import pandas as pd

from scipy.stats import randint

from datetime import datetime





    # Créez une fonction load_and_prep_data qui va charger les données à partir d'un fichier CSV et les préparer pour l'entraînement.a Chargement et préparation des données:
    # Cette fonction doit lire les données, séparer les caractéristiques (features) des étiquettes (labels), et diviser les données en ensembles d'entraînement et de validation.

    
def load_and_prep_data(data_file: str, data_location: str):
    ###
    # Loads and prepares data
    ###
    
    # load data
    data = pd.read_csv(f"{data_location}/{data_file}")
    
    # extract features
    X = data.drop(columns = ['date', 'demand'])
    X = X.astype('float')
    
    # extract target
    y = data.demand
    
    # Utilisez train_test_split de scikit-learn pour la division des données.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 321)
    
    return X_train, X_test, y_train, y_test


def setup_mlflow (experiment_name, uri):
    ###
    ###

    # Configurez le suivi avec l'URI de MLflow.
    mlflow.set_tracking_uri(uri)
    
    # Vérifiez si une expérience existe déjà avec ce nom et créez-la si nécessaire.
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_name = experiment_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = client.create_experiment(experiment_name)
    
    # Utilisez MlflowClient pour gérer les expériences et les exécutions.
    mlflow.set_experiment(experiment_name)    

    return client, expermiment_id


def main():
    ###
    ###
    
    # 1. Chargement et préparation des données:
    
    # load/define data location variables
    data_location = "data"
    data_file = "fake_data.csv"
    
    # load data_file from data_location and prepare
    X_train, X_test, y_train, y_test = load_and_prep_data(data_file, data_location)
    
    
    # 2. Configuration de MLflow:
    
    # load/define mlflow setup variables
    # Définissez un nom d'expérience pour MLflow.
    current_experiment_name = "challenge_experiment_1"
    tracking_uri = "http://127.0.0.1:8080"
    
    client, experiment_id = set_up_mlflow(current_experiment_name, tracking_uri)
    
    
    # 3. Activation de l'auto-enregistrement (autologging):

    # Activez l'autologging pour scikit-learn avec mlflow.sklearn.autolog.
    mlflow.sklearn.autolog(log_models = True)
    
    # 4. Définition de l'espace de recherche pour les hyperparamètres:

    # Définissez un dictionnaire des distributions de paramètres pour la recherche aléatoire (RandomizedSearchCV).
    # Utilisez randint de scipy.stats pour définir les plages de valeurs des hyperparamètres.
    param_ranges = {
        'n_estimators': randint(4, 2000),
        'max_depth': randint(5, 100),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1,4)
        }
    
    # 5. Recherche aléatoire des hyperparamètres:

    # Créez un modèle RandomForestRegressor.
    # Utilisez RandomizedSearchCV pour effectuer une recherche aléatoire sur les hyperparamètres.
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state = 321),
        param_distributions = param_ranges,
        n_iter = 5,
        cv = 5,
        scoring = 'r2',
        random_state = 321
    )
    
    # Entraînez le modèle avec les données d'entraînement.
    random_search.fit(X_train, y_train)
    
    
    # 6. Récupération des informations sur le meilleur modèle:

    # Récupérez les meilleurs hyperparamètres et le score de validation croisée (CV score) du meilleur modèle.
    best_params = random_search.best_params_
    
    # Utilisez la fonction de l'API python mlflow MlflowClient pour rechercher les exécutions et identifier celle ayant les meilleurs hyperparamètres (vous pouvez vous aider de la documentation en ligne).
    current_experiment_runs = client.search_runs(
        experiment_ids = [client.get_experiment_by_name(current_experiment_name).experiment_id],
        filter_string = ""
    )
    
    # parent_run = None
    # for run in current_experiment_runs:
    #     if 'best_n_estimators' in run.data.params:
    #         parent_run = run
    #         break
    
    parent_run = [run for run in current_experiment_runs if 'best_n_estimators' in run.data.params][0]
    
    best_params_from_parent = {
        'n_estimators': parent_run.data.params['best_n_estimators'],
        'max_depth': parent_run.data.params['best_max_depth'],
        'min_samples_split': parent_run.data.params['best_min_samples_split'],
        'min_samples_leaf': parent_run.data.params['best_min_samples_leaf']
    }
    
    best_run = None
    for run in current_experiment_runs:
        if ('n_estimators' in run.data.params and
            run.data.params['n_estimators'] == best_params_from_parent['n_estimators'] and
            run.data.params['max_depth'] == best_params_from_parent['max_depth'] and
            run.data.params['min_samples_split'] == best_params_from_parent['min_samples_split'] and
            run.data.params['min_samples_leaf'] == best_params_from_parent['min_samples_leaf']):
            best_run = run
            break
        
    best_run_name = best_run.data.tags.get('mlflow.runName', 'Not found') if best_run else 'Not found'
    
    
    # 7. Création d'un résumé des résultats:

    # Créez un résumé des résultats de la recherche aléatoire.
    summary = f"""Random Forest Trials Summary:
---------------------------
🏆 Best Experiment Name: {current_experiment}
🎯 Best Run Name: {best_run_name}

Best Model Parameters:
🌲 Number of Trees: {random_search.best_params_['n_estimators']}
📏 Max Tree Depth: {random_search.best_params_['max_depth']}
📎 Min Samples Split: {random_search.best_params_['min_samples_split']}
🍂 Min Samples Leaf: {random_search.best_params_['min_samples_leaf']}
📊 Best CV Score: {random_search.best_score_:.4f}
"""

    # Enregistrez ce résumé en tant qu'artifact dans MLflow.
    with mlflow.start_run(run_id=parent_run.info.run_id):

        # Log summary as an artifact
        with open("summary.txt", "w") as f:
            f.write(summary)
        mlflow.log_artifact("summary.txt")
        
    
    # 8. Exécution du script principal:

    # Créez une fonction main qui orchestre toutes les étapes ci-dessus.
    # Ajoutez un point d'entrée conditionnel pour exécuter main si le script est exécuté directement.

if __name__ == "__main__" :
    main()
    