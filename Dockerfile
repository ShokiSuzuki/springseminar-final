FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

WORKDIR /work

RUN apt update
RUN apt install -y tmux
RUN pip install matplotlib     

