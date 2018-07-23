#include <stdio.h>
#include "halloc.h"

__global__ void test() {
  void* x = hamalloc(128);
  printf("%p\n", x);
}

int main() {
  ha_init();
  test<<<10,5>>>();
cudaThreadSynchronize();
}
