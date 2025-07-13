from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
import joblib
from os.path import join as join

# variables de tunning
nbEpoche = 75
seed = 132
batch_size = 32
dataset = 'df_merge.csv'
test_size = 0.1

# Chargement des datasets
df_old = pd.read_csv(join('data',dataset))

# Charger le préprocesseur
# preprocessor_loaded = joblib.load(join('models','preprocessor.pkl'))

# preprocesser les data
X, y, _ = preprocessing(df_old)

# split data in train and test dataset
X_train, X_test, y_train, y_test = split(X, y, test_size=test_size)

# Chargement des datasets
# df_old = pd.read_csv(join('data','df_old.csv'))
# X_old, y_old, _ = preprocessing(df_old)
#
# X_train, X_test, y_train, y_test = split(X + X_old, y+y_old)

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


##retraining
##Chargement des datasets
# df_old = pd.read_csv(join('data','df_old.csv'))
# X, y, _ = preprocessing(df_old)
# X_train, X_test, y_train, y_test = split(X, y)
#
# model2, hist2 = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test, epochs=nbEpoche, verbose=1, batch_size=batch_size)
# y_pred = model_predict(model2, X_test)
# perf = evaluate_performance(y_test, y_pred)
# print_data(perf, exp_name="exp 2")
# draw_loss(hist2)



# y_pred = model_predict(model, X_test)
# perf = evaluate_performance(y_test, y_pred)
# print_data(perf, exp_name="exp 2")

exit(0)
# charger le modèle
model_2024_08 = joblib.load(join('models','model_2024_08.pkl'))

#%% predire sur les valeurs de train
y_pred = model_predict(model_2024_08, X_train)

# mesurer les performances MSE, MAE et R²
perf = evaluate_performance(y_train, y_pred)  

print_data(perf)

#%% predire sur les valeurs de tests
y_pred = model_predict(model_2024_08, X_test)

# mesurer les performances MSE, MAE et R²
perf = evaluate_performance(y_test, y_pred)   

print_data(perf)

#%% WARNING ZONE on test d'entrainer le modèle plus longtemps mais sur les mêmes données
model2, hist2 = train_model(model_2024_08, X_train, y_train, X_val=X_test, y_val=y_test)
y_pred = model_predict(model_2024_08, X_test)
perf = evaluate_performance(y_test, y_pred)
print_data(perf, exp_name="exp 2")
draw_loss(hist2)
