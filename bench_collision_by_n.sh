#!/bin/bash
for i in {0..15..1}
do
  /usr/local/cuda/bin/nvcc -I. -Iexample/configuration/soa_alloc -std=c++11 --expt-extended-lambda -gencode arch=compute_61,code=sm_61 -gencode arch=compute_50,code=sm_50 -maxrregcount=64 -O3 -DNDEBUG -Ilib/cub -DDEFRAG_FACTOR=${i} -o bench example/collision/soa/collision.cu
  ./bench > benchmark/collision_n_${i}.csv
  cat benchmark/collision_n_${i}.csv
done
