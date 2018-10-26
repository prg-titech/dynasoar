#!/bin/sh
/usr/local/cuda/bin/nvcc -std=c++11 -DNDEBUG -DHEAP_SIZE=500 -O3 -use_fast_math wa-tor/soa/wator.cu -DGRID_SIZE_Y=3300 -I. -arch compute_61 --expt-relaxed-constexpr
