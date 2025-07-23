import base64
import streamlit as st
import requests
import io
import os

from PIL import Image
from loguru import logger

from pydantic import BaseModel

#
# Setup APP
#

# BAckend URI
BACKEND_URI = os.getenv('BACKEND_URI', 'http://localhost:8000')

# Logging setup
logger.add("logs/frontend.log",
           level="TRACE",
           rotation="10 MB",
           retention="7d",
           compression="bz2",
           enqueue=True,
           format="{time} {level} {message}"
           )


def predict_UI():
    """
    Prediction UI
    """
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
        nombre_sinistres = st.number_input('Nombre sinistres', min_value=0, max_value=50, step=1)
        submited = st.form_submit_button(label='Prévision')
        if submited:
            try:
                logger.info(f"Prediction : called : region:{region}, type_contrat:{type_contrat}, age:{age}, anciennete_contrat:{anciennete_contrat}, nombre_sinistres:{nombre_sinistres}")
                # Validators
                if age < 0 or age > 120:
                    raise ValueError("L'âge doit être compris entre 0 et 120.")
                if anciennete_contrat < 0 or anciennete_contrat > 50:
                    raise ValueError("L'ancienneté du contrat doit être comprise entre 0 et 50.")
                if nombre_sinistres < 0 or nombre_sinistres > 50:
                    raise ValueError("Le nombre de sinistres doit être compris entre 0 et 50.")

                response = requests.post(f'{BACKEND_URI}/predict', json={
                    'region': region,
                    'type_contrat': type_contrat,
                    'age': age,
                    'anciennete_contrat': anciennete_contrat,
                    'nombre_sinistres': nombre_sinistres,
                })
                if response.status_code == 200:
                    st.success('Prediction successful')
                    montant = response.json().get('montant')
                    logger.info(f"Prediction : {montant}")
                    montant = round(montant, 2)
                    st.write(f"Montant prédit : {montant}")
                else:
                    st.error('Prediction failed')
                    logger.error(f"Prediction : {response.json()}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Prediction : Request to backend failed : {e}")
                st.error(f"Request failed: {e}")
            except ValueError as e:
                logger.error(f"Validation error: {e}")
                st.error(f"Validation error: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                st.error(f"An unexpected error occurred: {e}")


def train_UI():
    """
    Training UI
    """
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
            submitted = st.form_submit_button('Train')

            if submitted:
                try:
                    # Validation des entrées
                    if epochs < 1:
                        raise ValueError("Le nombre d'époques doit être supérieur ou égal à 1.")
                    if batch_size < 1:
                        raise ValueError("La taille du batch doit être supérieure ou égale à 1.")
                    if test_size < 0.01 or test_size > 0.99:
                        raise ValueError("La taille du test doit être comprise entre 0.01 et 0.99.")
                    if not run_name:
                        raise ValueError("Le nom du run ne peut pas être vide.")

                    logger.info(f"Training called : epochs:{epochs}, batch_size:{batch_size}, test_size:{test_size}, seed:{seed}, run_name:{run_name}")

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

                        if response.status_code == 200:
                            st.success('Training successful')
                            json_response = response.json()
                            image_bytes = base64.b64decode(json_response["loss"])
                            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                            st.image(image)
                        else:
                            st.error(f'Training failed with status code: {response.status_code}')
                            st.write(response.json())

                except requests.exceptions.RequestException as e:
                    logger.error(f"Request failed: {e}")
                    st.error(f"Request failed: {e}")
                except ValueError as e:
                    logger.error(f"Validation error: {e}")
                    st.error(f"Validation error: {e}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred: {e}")
                    st.error(f"An unexpected error occurred: {e}")

def health_UI():
    """
    Health UI
    """
    check = True
    if st.button('Health') or check:
        check = False
        try:
            logger.info("Health check called")
            response = requests.get(f'{BACKEND_URI}/health')
            if response.status_code == 200:
                st.success('Health check successful')
            else:
                st.error(f'Health check failed with status code: {response.status_code}')
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            st.error(f"Request failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            st.error(f"An unexpected error occurred: {e}")

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

