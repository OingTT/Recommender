FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
FROM tiagopeixoto/graph-tool:latest

COPY . .

RUN apt-get install python
