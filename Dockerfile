# FROM tiagopeixoto/graph-tool:latest
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY . .

# RUN apt -y -qq update &&\
#   apt install python
#   python -m pip install -r requirements.txt
