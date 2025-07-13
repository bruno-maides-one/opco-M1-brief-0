import datetime
import os
import mlflow
import tensorflow as tf

from modules.assurance_old.preprocess import preprocessing_old_dataset, preprocessing_new_dataset, split
from modules.evaluate import evaluate_performance
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
import joblib
from os.path import join as join
from tensorflow.keras.layers import Dense
from modules.backend import do_predict

# Init de MLFlow
# mlflow.autolog()
mlflow.set_tracking_uri(os.getenv('MLFLOW_URI', 'http://localhost:5000'))

model_old_params = {
    'epochs': 75,
    'batch_size': 25,
    'test_size': 0.2,
    'seed': 132,
    'dataset': 'ds_old.csv',
    'run_name': 'assurance_old'
}

model_new_params = {
    'epochs': 75,
    'batch_size': 25,
    'test_size': 0.2,
    'seed': 132,
    'dataset': 'ds_new.csv',
    'run_name': 'assurance_new'
}


def train_dataset_assurance(params, preprocess_function, creer_new=True, model=None, save_model=None, verbose=1):

    # Chargement des datasets
    df = pd.read_csv(join('data/assurance', params['dataset']))

    # preprocesser les data
    X, y, _ = preprocess_function(df)
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

# model_old, hist = train_dataset_assurance(model_old_params, preprocess_function=preprocessing_old_dataset)
# model_old, hist = train_dataset_assurance(model_old_params, preprocess_function=preprocessing_old_dataset, save_model="old_a_tester")

model_old = joblib.load(join('models','old_a_tester.pkl'))
model_old.summary()

do_predict(model_old,
        {
            'id_client': [1, ],
            'age': [10, ],
            'anciennete_contrat': [3, ],
            'nombre_sinistres': [1, ],
            'region': ['Bretagne', ],
            'type_contrat': ['Tiers', ],
            "montant_total_sinistres": [0, ],
        })
# df = pd.read_csv(join('data/assurance', 'ds_old.csv'))
# X, y, _ = preprocessing_old_dataset(df)


predict_data = pd.DataFrame({
    'id_client': [1,],
    'age': [10,],
    'anciennete_contrat': [3,],
    'nombre_sinistres': [1,],
    'region': ['Bretagne',],
    'type_contrat': ['Tiers',],
    "montant_total_sinistres": [0,],
})

predict_data = pd.read_csv(join('data/assurance', 'ds_old.csv'))[:20]

X, y, _ = preprocessing_old_dataset(predict_data)

print(X)

y_pred = model_predict(model_old, X)


prediction = do_predict(model_old, X)

print(prediction)

#
exit(0)

# Modification du model existant
for layer in model_old.layers:
    layer.trainable = True
model_old.add(Dense(32, activation='relu'))
model_old.compile(loss='mse', optimizer='adam')
model_old.summary()

model_new, hist = train_dataset_assurance(model_new_params, preprocess_function=preprocessing_new_dataset, model=model_old)
# # model_new, hist = train_dataset_assurance(model_new_params)
