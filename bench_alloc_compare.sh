#!/bin/bash
# Tested using bash version 4.1.5
for ((i=1;i<=1536;i++)); 
do 
  echo ${i}

  /usr/local/cuda/bin/nvcc -std=c++11 -DNUM_ALLOCS=${i} -DALLOC_SIZE=16 -O3 -use_fast_math microbench/linux_scalability_mallocmc.cu -I. -Imallocmc_copy/src/include -arch compute_61
  ./a.out >> bench_alloc_compare_mallocmc.csv
  echo "MALLOCMC"
  cat bench_alloc_compare_mallocmc.csv

  /usr/local/cuda/bin/nvcc -std=c++11 -DNUM_ALLOCS=${i} -DALLOC_SIZE=16 -O3 -use_fast_math microbench/linux_scalability_soa.cu -I. -arch compute_61
  ./a.out >> bench_alloc_compare_soa.csv

  echo "SOA"
  cat bench_alloc_compare_soa.csv


  /usr/local/cuda/bin/nvcc -std=c++11 -DNUM_ALLOCS=${i} -DALLOC_SIZE=16 -O3 -use_fast_math microbench/linux_scalability.cu -I. -arch compute_61
  ./a.out >> bench_alloc_compare_cuda.csv

  echo "CUDA"
  cat bench_alloc_compare_cuda.csv


  /usr/local/cuda/bin/nvcc -std=c++11 -DNUM_ALLOCS=${i} -DALLOC_SIZE=16 -O3 -use_fast_math microbench/linux_scalability_halloc.cu -I. -I/home/matthias/halloc/install/include -dc -arch compute_61 -o wator.o
  /usr/local/cuda/bin/nvcc -std=c++11 -O3 -use_fast_math -arch compute_61 -L/home/matthias/halloc/install/lib -lhalloc -o wator wator.o
  ./wator >> bench_alloc_compare_halloc.csv
echo "HALLOC"
cat bench_alloc_compare_halloc.csv

done
