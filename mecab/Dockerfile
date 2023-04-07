FROM trf_ner_base

WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     cmake \
#     file \
#     unzip \
#     gcc \
#     g++ \
#     xz-utils \
#     mecab \
#     libmecab-dev \
#     mecab-ipadic-utf8 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

COPY trf_train.py /app/trf_train.py

CMD ["python3", "/app/trf_train.py"]