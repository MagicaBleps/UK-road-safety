FROM python:3.10.6-buster
COPY uk_road_safety /uk_road_safety
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
RUN pip install --upgrade pip
RUN pip install .
CMD uvicorn uk_road_safety.API.api:app --host 0.0.0.0 --port $PORT
