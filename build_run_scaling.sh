#!/bin/bash

for i in `seq 0 256 8192`
do
  echo "Running ${i}"
  /usr/local/cuda/bin/nvcc -std=c++11 -O3 -use_fast_math wa-tor/aos/wator.cu -I. -arch compute_61 -I mallocmc_copy/src/include -DGRID_SIZE_X=256 -DGRID_SIZE_Y=${i}
#  /usr/local/cuda/bin/nvcc -std=c++11 -O3 -use_fast_math wa-tor/aos/wator_soa.cu -I. -arch compute_61 -DGRID_SIZE_X=256 -DGRID_SIZE_Y=${i}
  ./a.out >> bench_scaling_mallocmc.csv
  cat bench_scaling_mallocmc.csv
done
