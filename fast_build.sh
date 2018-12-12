#!/bin/sh
# To use clang: -ccbin=clang++-3.8

/usr/local/cuda/bin/nvcc -std=c++11 -O3 example/sugarscape/soa/sugarscape.cu example/sugarscape/soa/rendering.cu -I. -Iexample/configuration -Iexample/sugarscape/soa -arch compute_50 --expt-relaxed-constexpr -maxrregcount=64 -o bin/sugarscape_soa -lSDL2
