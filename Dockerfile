# from python 3.11
FROM python:3.11-slim-bookworm AS builder
LABEL authors="bruno"

# application takes place in /app
WORKDIR /app

# execute pip install with --no-cache-dir to avoid pip cache include final build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie application, check .dockerignore for excluded files
COPY . .

FROM builder AS backend

# Expose uvicorn port
EXPOSE 8000

# run backend using uvicorn on 0.0.0.0 to allow connection from any source
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]


FROM builder AS frontend

# Expose streamlit port
EXPOSE 8501

# run backend using uvicorn on 0.0.0.0 to allow connection from any source
CMD ["streamlit", "run", "frontend.py"]

FROM builder AS mlflow

# Expose uvicorn port
EXPOSE 5000

# run backend using uvicorn on 0.0.0.0 to allow connection from any source
CMD ["mlflow", "ui", "--host", "0.0.0.0"]
