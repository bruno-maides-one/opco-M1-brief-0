import os
import mlflow
import tensorflow as tf
import joblib

from modules.assurance_old.preprocess import preprocessing_old_dataset, preprocessing_new_dataset, split
from modules.evaluate import evaluate_performance
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
from os.path import join as join
from loguru import logger
from tensorflow.keras.layers import Dense


def load_model(model_name):
    """
    Loads the trained model
    """
    try:
        logger.debug("current directory" + os.getcwd())
        model_path = os.path.join('models', model_name)
        model = joblib.load(model_path)
        logger.info(f"Model '{model_name}' loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error(f"Error: The model file '{model_name}' was not found.")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading the model: {e}")
        return None

def save_model(model, model_name):
    """
    Saves the trained model
    """
    try:
        model_path = os.path.join('models', f'{model_name}.pkl')
        joblib.dump(model, model_path)
        logger.info(f"Model saved successfully as '{model_name}.pkl'.")
        return True
    except Exception as e:
        logger.error(f"An error occurred while saving the model: {e}")
        return False

def do_predict(model, params):
    """
    Do a prediction
    """
    try:
        df = pd.read_csv(join('data/assurance', 'ds_old.csv'), nrows=20)
        # dict = df.to_dict('list')
        # dict['id_client'].append(-1)
        # dict['montant_total_sinistres'].append(-1)
        # dict['age'].append(params['age'])
        # dict['anciennete_contrat'].append(params['anciennete_contrat'])
        # dict['nombre_sinistres'].append(params['nombre_sinistres'])
        # dict['region'].append(params['region'])
        # dict['type_contrat'].append(params['type_contrat'])
        # df = pd.DataFrame.from_dict(dict)
        logger.info(f"Predicting : age:{params['age']}, anciennete_contrat:{params['anciennete_contrat']}, nombre_sinistres:{params['nombre_sinistres']}, region:{params['region']}, type_contrat:{params['type_contrat']}")
        X, y, _ = preprocessing_old_dataset(df)
        prediction = model_predict(model, X)[-1]
        return prediction
    except FileNotFoundError:
        logger.error("Error: The dataset file 'ds_old.csv' was not found.")
        return None
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return None

def do_train(params, creer_new=True, model=None, save_model=None, verbose=1):
    """
    Trains the model
    """
    try:
        # Init de MLFlow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_URI', 'http://localhost:5000'))

        # Chargement des datasets
        df = pd.read_csv(params['dataset'])

        # preprocesser les data
        X, y, _ = preprocessing_old_dataset(df)
        # X, y, _ = preprocessing_old_dataset(df)

        # split data in train and test dataset
        X_train, X_test, y_train, y_test = split(X, y, test_size=params['test_size'], random_state=params['seed'])

        if creer_new and model is None:
            # create a new model, test if model is None is to prevent Mutable overwrite.
            # without this test if a model is provided it will be overwriten with empty model
            model = create_nn_model(X_train.shape[1])
        # else:
        #     model.trainable = True

        model.summary()

        # Model training
        with mlflow.start_run(run_name=params["run_name"]) as mlrun:
            # On log les parametres
            mlflow.log_params(params)
            # Model training
            model, hist = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=params['epochs'], verbose=verbose, batch_size=params['batch_size'])
            y_pred = model_predict(model, X_test)
            perf = evaluate_performance(y_test, y_pred)
            # Start mlflow tracking
            mlflow.log_metric("mse", perf['MSE'])
            mlflow.log_metric("mae", perf['MAE'])
            mlflow.log_metric("r2", perf['R²'])
            r_carre = perf["R²"]

        # sauvegarder le modèle
        if save_model is not None:
            # joblib.dump(model, join('models',f'model_assurance_old_R2-{params["r_carre"]}_epoche-{params["epoches"]}_seed-{params["seed"]}_batchsize-{params["batch_size"]}_testsize-{params["test_size"]}.pkl'))
            joblib.dump(model, join('models',f'{save_model}.pkl'))

        return model, hist
    except FileNotFoundError as fnf_error:
        logger.error(f"Error: The file was not found - {fnf_error}")
        return None, None
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        return None, None
