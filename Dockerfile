FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

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
    cmake \
    file \
    unzip \
    gcc \
    g++ \
    xz-utils \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc
RUN pip3 install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt
#     pip install hydra-core --upgrade


COPY run.sh /app/run.sh
# COPY config.yaml .
COPY data.py /app/data.py
COPY pl_vocabulary_trf.py /app/pl_vocabulary_trf.py
COPY pl_main.py /app/pl_main.py
COPY pl_module_trf.py /app/pl_module_trf.py
COPY pl_datamodule_trf.py /app/pl_datamodule_trf.py
# CMD ["bash", "/app/run.sh"]

COPY trf_main.py /app/trf_main.py
CMD ["python3", "/app/trf_main.py"]