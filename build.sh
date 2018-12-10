#!/bin/sh

# TODO: Make proper build file.
# To use clang: -ccbin=clang++-3.8

#/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 -use_fast_math example/wa-tor/soa/wator.cu -DGRID_SIZE_Y=512 -I. -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64 -o bin/wator
#/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 example/nbody/soa/nbody.cu example/nbody/soa/rendering.cu -I. -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64 -o bin/nbody -lSDL2
#/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 example/nbody/soa/nbody_baseline.cu example/nbody/soa/rendering.cu -I. -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64 -o bin/nbody_baseline -lSDL2
#/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 example/api-example/example.cu -I. -arch compute_61 --expt-relaxed-constexpr -maxrregcount=64 -o bin/api-example
#/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 example/game-of-life/soa/gol.cu example/game-of-life/soa/rendering.cu example/game-of-life/soa/dataset_loader.cu -I. -arch compute_50 --expt-relaxed-constexpr -maxrregcount=64 -o bin/gol -lSDL2
#/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 example/game-of-life/soa/gol_baseline.cu example/game-of-life/soa/rendering.cu example/game-of-life/soa/dataset_loader.cu -arch compute_50 -I. --expt-relaxed-constexpr -o bin/gol_baseline -lSDL2

/usr/local/cuda/bin/nvcc -std=c++11 -DHEAP_SIZE=500 -O3 example/sugarscape/soa/sugarscape.cu -I. -arch compute_50 --expt-relaxed-constexpr -maxrregcount=64 -o bin/sugarscape -lSDL2
