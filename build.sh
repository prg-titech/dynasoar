#!/bin/sh
#/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 -use_fast_math example/wa-tor/soa/wator.cu -DGRID_SIZE_Y=512 -I. -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64
/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 -use_fast_math example/bfs/bfs.cu -I. -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64
