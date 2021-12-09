#FROM gcr.io/kaggle-gpu-images/python:latest
#FROM nvcr.io/partners/gridai/pytorch-lightning:v1.4.0
FROM rapidsai/rapidsai:cuda11.4-base-ubuntu20.04-py3.8

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY ./ ./

RUN apt update && apt -y upgrade && apt install -y \
  build-essential \
  cmake \
  git \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev \
  libgl1-mesa-dev

# Install Library
RUN pip install --upgrade pip
# Install PyTorch
#RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#RUN pip install -r requirements.txt


# Install LightGBM
#RUN git clone --recursive https://github.com/microsoft/LightGBM && cd LightGBM \
#  && mkdir build \
#  && cd build \
#  && cmake .. \
#  && make -j4
#
#RUN cd LightGBM/python-package \
#  && python setup.py install
#
#RUN rm -r -f LightGBM/