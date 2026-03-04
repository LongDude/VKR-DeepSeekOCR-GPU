FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
WORKDIR /

# Стандартные утилиты и update
RUN apt-get update --yes && \
    apt-get upgrade --yes --no-install-recommends \
    git wget curl
RUN apt-get install --yes --no-install-recommends \
    software-properties-common

# Python 3.11 
RUN add-apt-repository 'ppa:deadsnakes/ppa'
RUN apt-get update && apt-get install --yes --no-install-recommends\
    python3.11 python3.11-venv python3.11-dev python3-pip

# Чистка
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

WORKDIR /app