FROM nvidia/cuda:11.4.1-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive 

RUN apt-get update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools

RUN pip3 -q install pip --upgrade

WORKDIR /home
COPY requirements.txt /home/
RUN pip install -r requirements.txt
RUN pip install psycopg2-binary

COPY src /home/src