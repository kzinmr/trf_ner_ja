FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV APP_ROOT /app
ENV WORK_DIR /app/workspace

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y \
    build-essential \
    sudo \
    git \
    wget \
    curl \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

COPY span2conll.py /app/span2conll.py
