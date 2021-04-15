#!/bin/bash

docker run -it --rm --gpus all \
    -v ${HOME}:/home/work/ \
    --name springseminar springseminar/pytorch:1.8.0-cuda11.1
