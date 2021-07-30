FROM pytorch/pytorch

ENV DEBIAN_FRONTEND noninteractive 

RUN apt-get update

WORKDIR /home
COPY requirements.txt /home/
RUN pip install -r requirements.txt
RUN pip install psycopg2-binary

COPY src /home/src

