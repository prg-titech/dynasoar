#!/bin/sh
/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 -use_fast_math wa-tor/aos/wator_soa.cu -DGRID_SIZE_Y=512 -I. -arch compute_61 --expt-relaxed-constexpr
