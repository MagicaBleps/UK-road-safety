FROM python:3.10.6-buster
COPY . .
RUN pip install --upgrade pip
RUN pip install .
CMD uvicorn uk_road_safety.API.api:app --host 0.0.0.0 --port $PORT
