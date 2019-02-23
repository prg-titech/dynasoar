#!/bin/bash
for i in {1..15..1}
do
  echo ${i}
  /usr/local/cuda/bin/nvcc -I. -Iexample/configuration/soa_alloc -std=c++11 --expt-extended-lambda -gencode arch=compute_61,code=sm_61 -gencode arch=compute_50,code=sm_50 -maxrregcount=64 -O3 -DNDEBUG -Ilib/cub -DDEFRAG_FACTOR=${i} -o bench example/wa-tor/soa/wator.cu example/wa-tor/rendering.cu -lSDL2
  echo Running...
  ./bench > benchmark/wator_n_${i}.csv
  cat benchmark/wator_n_${i}.csv
done
