/usr/local/cuda/bin/nvcc -std=c++11 -O3 -use_fast_math wa-tor/aos/wator.cu -I. -I/home/matthias/halloc/install/include -dc -arch compute_50 -o wator.o

/usr/local/cuda/bin/nvcc -std=c++11 -O3 -use_fast_math -arch compute_50 -L/home/matthias/halloc/install/lib -lhalloc -lSDL2 -lSDL2_gfx -o wator wator.o

