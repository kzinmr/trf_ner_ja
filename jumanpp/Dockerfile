FROM trf_ner_base

WORKDIR /app

# ARG JPP_VERSION=2.0.0-rc3

# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     make \
#     cmake \
#     libprotobuf-dev \
#     ca-certificates \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# RUN wget https://github.com/ku-nlp/jumanpp/releases/download/v${JPP_VERSION}/jumanpp-${JPP_VERSION}.tar.xz -qO - \
#     | tar Jxf - \
#     && cd jumanpp-${JPP_VERSION} \
#     && mkdir bld \
#     && cd bld \
#     && cmake .. -DCMAKE_BUILD_TYPE=Release \
#     && make -j $([ $(nproc) -le 8 ] && echo "$(nproc)" || echo "8") \
#     && make install

# Configure Japanese locale
RUN apt-get update -q && apt-get install -yq \
    locales \
    && locale-gen ja_JP.UTF-8 \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/*
ENV LANG="ja_JP.UTF-8" \
    LANGUAGE="en_US" \
    LC_ALL="ja_JP.UTF-8"
RUN localedef -f UTF-8 -i ja_JP ja_JP.utf8

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

COPY trf_train.py /app/trf_train.py

CMD ["python3", "/app/trf_train.py"]