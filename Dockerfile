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


RUN apt-get update && apt-get install -y \
    cmake \
    file \
    unzip \
    gcc \
    g++ \
    xz-utils \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc

ARG JPP_VERSION=2.0.0-rc3

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    libprotobuf-dev \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/ku-nlp/jumanpp/releases/download/v${JPP_VERSION}/jumanpp-${JPP_VERSION}.tar.xz -qO - \
    | tar Jxf - \
    && cd jumanpp-${JPP_VERSION} \
    && mkdir bld \
    && cd bld \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j $([ $(nproc) -le 8 ] && echo "$(nproc)" || echo "8") \
    && make install

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

COPY data.py /app/data.py
COPY span2conll.py /app/span2conll.py
