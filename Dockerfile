FROM python:3.8

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts .

ENV TRANSFORMERS_CACHE=/models/
ENV TRANSFORMERS_OFFLINE='1'