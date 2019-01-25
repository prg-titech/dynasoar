#ifndef UTIL_UTIL_H
#define UTIL_UTIL_H

#include <type_traits>

#define __DEV__ __device__
#define __DEV_HOST__ __device__ __host__

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Custom std::array because std::array is not available on device.
template<typename T, size_t N>
class DeviceArray {
 private:
  T data[N];

 public:
  using BaseType = T;

  __device__ T& operator[](size_t pos) {
    return data[pos];
  }

  __device__ const T& operator[](size_t pos) const {
    return data[pos];
  }
};

// Check if type is a device array.
template<typename>
struct is_device_array : std::false_type {};

template<typename T, size_t N>
struct is_device_array<DeviceArray<T, N>> : std::true_type {};

// Reads value at a device address and return it.
template<typename T>
T read_from_device(T* ptr) {
  T host_storage;
  cudaMemcpy(&host_storage, ptr, sizeof(T), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
  return host_storage;
}

// A wrapper that runs a device member function.
template<typename C, void (C::*func)()>
__global__ void member_func_kernel(C* ptr) {
  (ptr->*func)();
}

template<typename C, typename T1, void (C::*func)(T1)>
__global__ void member_func_kernel(C* ptr, T1 t1) {
  (ptr->*func)(t1);
}

template<typename C, typename T1, typename T2, void (C::*func)(T1, T2)>
__global__ void member_func_kernel(C* ptr, T1 t1, T2 t2) {
  (ptr->*func)(t1, t2);
}

#endif  // UTIL_UTIL_H
