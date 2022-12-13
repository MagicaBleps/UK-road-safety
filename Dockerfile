#FROM python:3.10.6-slim
FROM tensorflow/tensorflow:2.11.0
COPY . .
RUN pip install --upgrade pip
RUN pip install .
CMD uvicorn uk_road_safety.API.api:app --host 0.0.0.0 --port $PORT
