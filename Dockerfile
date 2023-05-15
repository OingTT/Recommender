FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get -y update && apt-get -y upgrade

RUN apt-get install -y \
  python3-pip \
  software-properties-common \
  python3-gv \
  libcairo2-dev \
  libxt-dev \
  libgirepository1.0-dev \
  libboost1.74 \
  libpython3.11

# Fix "Unmet dependencies" => https://www.baeldung.com/linux/unmet-dependencies-apt-get
RUN apt-get install -f &&\
  pip install matplotlib pycairo pygobject &&\
  add-apt-repository "deb [ arch=amd64 ] https://downloads.skewed.de/apt bookworm main" &&\
  apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25 &&\
  apt-get update &&\
  apt-get install python3-graph-tool

COPY ./requirements.txt ./

RUN pip install -r requirements.txt &&\
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY . .