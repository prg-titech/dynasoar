#!/bin/sh
#/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 -use_fast_math example/wa-tor/soa/wator.cu -DGRID_SIZE_Y=512 -I. -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64 -o bin/wator
#/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 -use_fast_math example/nbody/soa/nbody.cu -I. -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64 -o bin/nbody
/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 -use_fast_math example/nbody/soa/nbody_baseline.cu example/nbody/soa/rendering.cu -I. -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64 -o bin/nbody_baseline -lSDL2
