import numpy

from modules.backend import load_model, save_model, do_predict, do_train

train_test_params = {
    'dataset': 'ds_old.csv',
    'test_size': 0.2,
    'seed': 42,
    'epochs': 10,
    'batch_size': 32,
    'run_name': 'test_run'
}

predict_test_params = {
    'age': 52,
    'anciennete_contrat': 3,
    'nombre_sinistres': 1,
    'region': 'Bretagne',
    'type_contrat': 'Tiers',
}

def test_load_model():
    model = load_model('old_a_tester.pkl')
    assert model is not None

def test_save_model():
    model = load_model('old_a_tester.pkl')
    saved = save_model(model, 'model_to_trash')
    assert saved is True

def test_do_predict():
    model = load_model('old_a_tester.pkl')
    prediction = do_predict(model, predict_test_params)
    assert prediction is not None
    assert isinstance(prediction, numpy.float32)
