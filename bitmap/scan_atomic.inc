// Textual header for inclusion in bitmap.h.

__DEV__ void atomic_add_scan_init() {
  enumeration_result_size = 0;
}

__DEV__ void atomic_add_scan() {
  SizeT* selected = nested.data_.enumeration_result_buffer;
  SizeT num_selected = nested.data_.enumeration_result_size;
  //printf("num_selected=%i\n", (int) num_selected);

  for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
       sid < num_selected; sid += blockDim.x * gridDim.x) {
    SizeT container_id = selected[sid];
    auto value = containers[container_id];
    int num_bits = __popcll(value);

    auto before = atomicAdd(reinterpret_cast<unsigned int*>(&enumeration_result_size),
                            num_bits);

    for (int i = 0; i < num_bits; ++i) {
      int next_bit = __ffsll(value) - 1;
      assert(next_bit >= 0);
      enumeration_result_buffer[before+i] = container_id*kBitsize + next_bit;
      //Advance to next bit.
      value &= value - 1;
    }
  }      
}

static const int kNumBlocksAtomicAdd = 256;

void run_atomic_add_scan() {
  nested.scan();

  SizeT num_selected = read_from_device<SizeT>(&nested.data_.enumeration_result_size);
  member_func_kernel<ThisClass, &ThisClass::atomic_add_scan_init><<<1, 1>>>(this);
  gpuErrchk(cudaDeviceSynchronize());
  member_func_kernel<ThisClass, &ThisClass::atomic_add_scan>
      <<<(num_selected + kNumBlocksAtomicAdd - 1)/kNumBlocksAtomicAdd,
         kNumBlocksAtomicAdd>>>(this);
  gpuErrchk(cudaDeviceSynchronize());
}
