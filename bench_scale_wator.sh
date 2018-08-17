#!/bin/bash
# Tested using bash version 4.1.5
for ((i=1;i<=768;i+=1)); 
do 
  echo ${i}

  /usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=${i} -O3 -use_fast_math wa-tor/aos/wator_soa.cu -I. -Icub-1.8.0 -arch compute_61
  ./a.out >> bench_wator_scale_enum_hierarchy_add.csv

  echo "bench_wator_scale_enum_hierarchy_add"
  cat bench_wator_scale_enum_hierarchy_add.csv



done
