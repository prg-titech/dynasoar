#!/bin/bash
for i in {0..100..1}
do
  /usr/local/cuda-9.1/bin/nvcc -I. -std=c++11 --expt-extended-lambda -gencode arch=compute_61,code=sm_61 -gencode arch=compute_50,code=sm_50 -maxrregcount=64 -O3 -DNDEBUG -Ilib/cub -DDELETE_RATIO=${i} -o bench example/defrag-bench/defrag_benchmark.cu
  ./bench > benchmark/n_5_delete_${i}.csv
  cat benchmark/n_5_delete_${i}.csv
done
