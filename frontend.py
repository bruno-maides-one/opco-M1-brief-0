import base64
import streamlit as st
import requests
import io
import os

from PIL import Image
from pydantic import BaseModel

BACKEND_URI = os.getenv('BACKEND_URI', 'http://localhost:8000')

def predict_UI():
    st.header('Assurance Prediction')
    with (st.form(key='predict-form')):
        region = st.selectbox('Region', (
            "Auvergne-Rhône-Alpes",
            "Nouvelle-Aquitaine",
            "Bretagne",
            "Provence-Alpes-Côte d'Azur",
            "Île-de-France"
        ))
        type_contrat = st.selectbox("Type de contrat", ("Tiers", "Tous risques"))
        age = st.number_input('Age', min_value=0, max_value=120, step=1, value=18)
        anciennete_contrat = st.number_input('Anciennete Contrat', min_value=0, max_value=50, step=1)
        nombre_sinistre = st.number_input('Nombre sinistres', min_value=0, max_value=50, step=1)
        submited = st.form_submit_button(label='Prévision')
        if submited:
            response = requests.post(f'{BACKEND_URI}/predict', json={
                'region': region,
                'type_contrat': type_contrat,
                'age': age,
                'anciennete_contrat': anciennete_contrat,
                'nombre_sinistre': nombre_sinistre,
            })
            if response.status_code == 200:
                st.success('Prediction successful')
            else:
                st.error('Prediction failed')
            st.write(response.json())


def train_UI():
    st.header('Assurance Training')
    uploaded_file = st.file_uploader("Selectionner un fichier source", type="csv")
    if uploaded_file is not None:
        st.write(f"Nom: {uploaded_file.name}")
        st.write(f"Type: {uploaded_file.type}")
        st.write(f"Taille: {uploaded_file.size} bytes")
        with st.form(key='train-form'):
            epochs = st.number_input("Epochs", value=75, min_value=1, step=1)
            batch_size = st.number_input("Batch size", value=25, min_value=1, step=1)
            test_size = st.number_input("Test size", value=0.2, min_value=0.01, max_value=0.99, step=0.01)
            seed = st.number_input("Seed", value=123, step=1)
            run_name = st.text_input("run name")
            submited = st.form_submit_button('Train')
            if submited:
                base64_csv = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                with st.spinner('Training...'):
                    response = requests.post(f"{BACKEND_URI}/train/", json={
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'test_size': test_size,
                        'seed': seed,
                        'dataset': base64_csv,
                        'run_name': run_name,
                    })
                    json_response = response.json()
                    image_bytes = base64.b64decode(json_response["loss"])
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    st.image(image)


def health_UI():
    response = requests.get(f'{BACKEND_URI}/health')
    if response.status_code == 200:
        st.success('Health check successful')
        st.write(response.json())
    else:
        st.error('Health check failed')


#
# Interface
#
#
# Frontend interface
#
st.title('Machin avec un model')

tab1, tab2, tab3 = st.tabs(["Predict", "Train", "Health"])

with tab1:
    predict_UI()
with tab2:
    train_UI()
with tab3:
    health_UI()

