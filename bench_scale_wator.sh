#!/bin/bash
# Tested using bash version 4.1.5
for ((i=1;i<=4096;i+=8)); 
do 
  echo ${i}

  /usr/local/cuda/bin/nvcc -std=c++11 -DGRID_SIZE_Y=${i} -O3 -use_fast_math wa-tor/aos/wator_soa.cu -I. -arch compute_61
  ./a.out >> bench_wator_scale_soa_enum_vw.csv

  echo "SOA"
  cat bench_wator_scale_soa_enum_vw.csv



done
