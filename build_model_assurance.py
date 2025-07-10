import datetime
import os

import mlflow


from modules.assurance_old.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
import joblib
from os.path import join as join

# variables de tunning
nbEpoche = 75
seed = 132
batch_size = 12
dataset = 'ds_old.csv'
test_size = 0.2

model_params: dict[str, int | float] = {
    'epochs': nbEpoche,
    'batch_size': batch_size,
    'test_size': test_size,
    'seed': seed,
}

mlflow.autolog()
mlflow.set_tracking_uri(os.getenv('MLFLOW_URI', 'http://localhost:5000'))

# Chargement des datasets
df = pd.read_csv(join('data/assurance', dataset))

# preprocesser les data
X, y, _ = preprocessing(df)

# split data in train and test dataset
X_train, X_test, y_train, y_test = split(X, y, test_size=test_size)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # create a new model 
model = create_nn_model(X_train.shape[1])

# entraîner le modèle
with mlflow.start_run(run_name="assurance_old") as mlrun:
    model, hist = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=nbEpoche, verbose=1, batch_size=batch_size)
    y_pred = model_predict(model, X_test)
    perf = evaluate_performance(y_test, y_pred)
    # mlflow.log_model_params(model_params, model_id="pouet")
    mlflow.log_metric("mse", perf['MSE'])
    mlflow.log_metric("mae", perf['MAE'])
    mlflow.log_metric("r2", perf['R²'])
    r_carre = perf["R²"]

    # draw_loss(hist)


# sauvegarder le modèle
datetimefile = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
joblib.dump(model, join('models',f'model_assurance_old_R2-{r_carre}_epoche-{nbEpoche}_seed-{seed}_batchsize-{batch_size}_testsize-{test_size}.pkl'))
