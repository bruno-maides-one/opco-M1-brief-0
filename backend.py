import base64

from fastapi import FastAPI
from modules.backend import load_model, do_predict, do_train
from pydantic import BaseModel
from io import StringIO, BytesIO
import matplotlib.pyplot as plt

def loss_graf(history):
    """
    Affiche les courbes de loss et val_loss de l'historique d'entraînement.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss (Entraînement)')
    plt.plot(history.history['val_loss'], label='Val Loss (Validation)', linestyle='--')
    plt.title('Courbes de Loss et Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Sauvegarder la figure dans un buffer en mémoire
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    # Convertir le buffer en chaîne base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return image_base64



class Prediction_params(BaseModel):
    region: str
    type_contrat: str
    age: int
    anciennete_contrat: int
    nombre_sinistres: int

class Training_params(BaseModel):
    epochs: int
    batch_size: int
    test_size: float
    seed: int
    dataset: str
    run_name: str

# Init FastAPI
app = FastAPI()
model = load_model('old_a_tester.pkl')

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(data: Prediction_params):
    dict = data.model_dump()
    prediction = float(do_predict(model, dict))
    return {'montant': prediction}

@app.get("/health")
async def health():
    return {'status': 'healthy'}

@app.post("/train")
async def train(data: Training_params):
    content_csv = base64.b64decode(data.dataset).decode('utf-8')
    csv_file_like = StringIO(content_csv)
    params = {
        'dataset': csv_file_like,
        'test_size': data.test_size,
        'seed': data.seed,
        'epochs': data.epochs,
        'batch_size': data.batch_size,
        'run_name': data.run_name,
    }

    model, hist = do_train(params)
    return {
        'loss': loss_graf(hist)
    }

