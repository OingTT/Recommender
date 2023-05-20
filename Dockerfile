FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set Timezone to Asia/Seoul
ENV TZ=Asia/Seoul
RUN apt-get -y update &&\
  apt-get -y upgrade &&\
  apt-get -yq install tzdata &&\
  ln -snf /usr/share/zoneinfo/$TZ /etc/localtime &&\
  echo $TZ > /etc/timezone &&\
  dpkg-reconfigure -f noninteractive tzdata

# Install essential packages
RUN apt-get install -y \
  software-properties-common \
  python3-software-properties

# Change python version to 3.11
RUN add-apt-repository ppa:deadsnakes/ppa &&\
  apt-get install -y \
  python3.11 \
  python3.11-distutils \
  curl \
  wget &&\
  update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 &&\
  curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Install Anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh &&\
  bash Anaconda3-2023.03-1-Linux-x86_64.sh -b &&\
  rm Anaconda3-2023.03-1-Linux-x86_64.sh &&\
  echo "export PATH=/root/anaconda3/bin:$PATH" >> ~/.bashrc &&\
  /root/anaconda3/bin/conda init bash &&\
  /root/anaconda3/bin/conda init zsh &&\
  # echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.temp &&\
  echo "conda create -y --name recommender python=3.10" >> ~/.temp &&\
  echo "conda activate recommender" >> ~/.temp &&\
  echo "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia" >> ~/.temp &&\
  echo "conda install graph-tool -c conda-forge" >> ~/.temp

RUN . ~/.bashrc && . ~/.temp


COPY ./requirements.txt ./requirements.txt

RUN /root/anaconda3/envs/recommender/bin/python -m pip install -r ./requirements.txt

# Copy file
COPY . .

ENTRYPOINT ["/root/anaconda3/envs/recommender/bin/python", "GHRS_test.py", "--debug", "--latent_dim", "8", "--max_epoch", "20"]

# RUN /bin/bash conda activate recommender

# Install requirements
# RUN /bin/bash -c "python -m pip install -r requirements.txt" &&\
#   /bin/bash -c "python -m pip install wandb pytorch-lightning pandas scikit-learn mysql-connector-python python-dotenv"

# # Install Graph-tool via conda
# RUN conda create -y --name recommender python=3.10 -c conda-forge graph-tool &&\
#   conda activate recommender &&\
#   conda list -e > requirements.txt

# # Install Graph-Tool
# RUN apt-get install -f &&\
#   python -m pip install --upgrade pip setuptools wheel &&\
#   apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25 &&\
#   add-apt-repository "deb https://downloads.skewed.de/apt buster main" &&\
#   apt-get update &&\
#   apt-get -t buster -yf install python3-graph-tool

# # Install python packages and pytorch
# COPY ./requirements.txt ./
# RUN python -m pip uninstall -y scipy &&\
#   python -m pip install -r requirements.txt &&\
#   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# # Install Boost
# # Todo: 아직 Build 안해봄
# # apt-get install -y aptitude =. 해도 안됨
# # apt install -y --fix-missing libboost-all-dev 하면 설치는 됨
# # 근데 오류는 계속 남음
# RUN  apt purge -y libboost-all-dev &&\
#   apt -y update &&\
#   apt -y upgrade &&\
#   apt install -y --fix-missing \
#   libboost-all-dev &&\
#   aptitude install -y libboost1.67-all-dev
# # sudo apt -o Debug::pkgProblemResolver=yes dist-upgrade
# # aptitude install -y libboost1.67-all-dev

# COPY . .