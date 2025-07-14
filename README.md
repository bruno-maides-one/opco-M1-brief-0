# Model training and prediction

## Development

In development mode, the system is designed to be executed without `.env` file.

### Backend

simply execute backend in reload mode in a console in order to update on file modification

```shell
uvicorn backend:app --reload
```

**The backend will be availlable at http://localhost:8000**

### Frontend

simply execute frontend in reload mode in a console in order to update on file modification

```shell
streamlit run frontend.py
```
**The front will be availlable at http://localhost:8501**

### MLFlow

MFlow service will be launch to manage training tracking, only tracking feature will be use in this exercice.

```shell
mlflow ui
```

**MLflow will be availlable at http://localhost:5000**


## Production

The solution is designed to be deployed using docker through a docker-compose serving the 3 services in a stack.
For the purpose of this exercice, docker will be used in local mode, there is no registry nor container manager such as docker-swarm or kubernetes

### Deployment

Using the `docker-compose.yaml` file deployment can by realized with the command

```shell
docker compose up -d
```


The stack enclose backend and only frontend and mlflow will be exposed.
* **frontend**: http://localhost:3000
* **MLFlow** : http://localhost:5000

### Build

In case of source update, stack need to be rebuild to take in account modification.

```shell
# stop the stack
docker compose down
# update the stack with current source
docker compose build
# run the updated stack
docker compose up -d
```
