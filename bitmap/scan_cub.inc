// Textual header for inclusion in bitmap.h.

// TODO: We probably do not need all these buffers.
SizeT enumeration_base_buffer[NumContainers*kBitsize];
SizeT enumeration_id_buffer[NumContainers*kBitsize];
SizeT enumeration_cub_temp[3*NumContainers*kBitsize];
SizeT enumeration_cub_output[NumContainers*kBitsize];
    
__DEV__ void set_result_size() {
  SizeT num_selected = nested.data_.enumeration_result_size;
  // Base buffer contains prefix sum.
  SizeT result_size = enumeration_cub_output[num_selected*kBitsize - 1];
  enumeration_result_size = result_size;
}

// TODO: Run with num_selected threads, then we can remove the loop.
__DEV__ void pre_scan() {
  SizeT* selected = nested.data_.enumeration_result_buffer;
  SizeT num_selected = nested.data_.enumeration_result_size;

  for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
       sid < num_selected; sid += blockDim.x * gridDim.x) {
    SizeT container_id = selected[sid];
    auto value = containers[container_id];
    for (int i = 0; i < kBitsize; ++i) {
      // Write "1" if allocated, "0" otherwise.
      bool bit_selected = value & (static_cast<ContainerT>(1) << i);
      enumeration_base_buffer[sid*kBitsize + i] = bit_selected;
      if (bit_selected) {
        enumeration_id_buffer[sid*kBitsize + i] =
            kBitsize*container_id + i;
      }
    }
  }
}

// Run with num_selected threads.
// Assumption: enumeration_base_buffer contains exclusive prefix sum.
__DEV__ void post_scan() {
  SizeT* selected = nested.data_.enumeration_result_buffer;
  SizeT num_selected = nested.data_.enumeration_result_size;

  for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
       sid < num_selected; sid += blockDim.x * gridDim.x) {
    SizeT container_id = selected[sid];
    auto value = containers[container_id];
    for (int i = 0; i < kBitsize; ++i) {
      // Write "1" if allocated, "0" otherwise.
      bool bit_selected = value & (static_cast<ContainerT>(1) << i);
      if (bit_selected) {
        // Minus one because scan operation is inclusive.
        enumeration_result_buffer[enumeration_cub_output[sid*kBitsize + i] - 1] =
            enumeration_id_buffer[sid*kBitsize + i];
      }
    }
  }
}

void run_cub_scan() {  
  nested.scan();

  SizeT num_selected = read_from_device<SizeT>(&nested.data_.enumeration_result_size);
  member_func_kernel<ThisClass, &ThisClass::pre_scan>
      <<<num_selected/256+1, 256>>>(this);
  gpuErrchk(cudaDeviceSynchronize());

  size_t temp_size = 3*NumContainers*kBitsize;
  cub::DeviceScan::InclusiveSum(enumeration_cub_temp,
                                temp_size,
                                enumeration_base_buffer,
                                enumeration_cub_output,
                                num_selected*kBitsize);
  member_func_kernel<ThisClass, &ThisClass::post_scan>
      <<<num_selected/256+1, 256>>>(this);
  gpuErrchk(cudaDeviceSynchronize());
  member_func_kernel<ThisClass, &ThisClass::set_result_size><<<1, 1>>>(this);
  gpuErrchk(cudaDeviceSynchronize());
}
