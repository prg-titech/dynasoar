#!/bin/sh
/usr/local/cuda/bin/nvcc -std=c++11 -O3 -arch compute_50 bench1.cu -I.
