import datetime

from modules import preprocessing, split
from modules import evaluate_performance
from modules import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
import joblib

from os.path import join as join
from loguru import logger

# variables de tunning
nbEpoche = 75
seed = 132
batch_size = 32
dataset = 'df_merge.csv'
test_size = 0.1

logger.debug(f'epochs: {nbEpoche}, seed: {seed}, batch_size: {batch_size}, dataset: {dataset}, test_size: {test_size}')

# Chargement des datasets
df = pd.read_csv(join('data', dataset))

# preprocesser les data
X, y, _ = preprocessing(df)

# split data in train and test dataset
X_train, X_test, y_train, y_test = split(X, y, test_size=test_size)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # create a new model
model = create_nn_model(X_train.shape[1])

# # entraîner le modèle
model, hist = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=nbEpoche, verbose=1, batch_size=batch_size)
draw_loss(hist)

y_pred = model_predict(model, X_test)
perf = evaluate_performance(y_test, y_pred)
print_data(perf, exp_name="exp 2")
r_carre = perf["R²"]

# # sauvegarder le modèle
datetimefile = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
joblib.dump(model, join('models',f'model_bruno_R2_{r_carre}_{datetimefile}.pkl'))

#
# Best model ever
#
# model = joblib.load(join('models','model_bruno_R2_0.9018189840850476_20250708.154255.pkl'))



