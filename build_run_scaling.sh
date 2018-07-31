#!/bin/bash

for i in `seq 0 256 65536`
do
  /usr/local/cuda/bin/nvcc -std=c++11 -O3 -use_fast_math wa-tor/aos/wator_soa.cu -I. -arch compute_61 -DGRID_SIZE_X=256 -DGRID_SIZE_Y=${i}
  ./a.out >> bench_scaling.csv
done
