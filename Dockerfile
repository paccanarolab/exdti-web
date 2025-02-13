FROM tiangolo/uvicorn-gunicorn-fastapi

COPY ./ /app

RUN pip install -r /app/requirements.txt

WORKDIR /app