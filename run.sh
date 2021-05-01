#!/bin/bash

docker run -it --rm --gpus all \
    -v ${PWD}:/work \
    --name springseminar springseminar/pytorch:1.8.0-cuda11.1
