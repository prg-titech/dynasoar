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
  T data_[N];

 public:
  using BaseType = T;

  __device__ T& operator[](size_t pos) {
    assert(pos < N);
    return data_[pos];
  }

  __device__ const T& operator[](size_t pos) const {
    assert(pos < N);
    return data_[pos];
  }

  __device__ volatile T& operator[](size_t pos) volatile {
    assert(pos < N);
    volatile T* data_ptr = data_ + pos;
    return *data_ptr;
  }

  __device__ const volatile T& operator[](size_t pos) const volatile {
    assert(pos < N);
    const volatile T* data_ptr = data_ + pos;
    return *data_ptr;
  }

  template<typename U = T>
  __device__ typename std::enable_if<sizeof(U) == 4, U>::type
  atomic_cas(size_t pos, T assumed, T value) {
    assert(pos < N);

    auto* ptr_assumed = reinterpret_cast<unsigned int*>(&assumed);
    auto* ptr_value = reinterpret_cast<unsigned int*>(&value);
    auto* ptr_addr = reinterpret_cast<unsigned int*>(data_ + pos);

    auto result = atomicCAS(ptr_addr, *ptr_assumed, *ptr_value);
    return *reinterpret_cast<U*>(&result);
  }

  template<typename U = T>
  __device__ typename std::enable_if<sizeof(U) == 8, U>::type
  atomic_cas(size_t pos, T assumed, T value) {
    assert(pos < N);

    auto* ptr_assumed = reinterpret_cast<unsigned long long int*>(&assumed);
    auto* ptr_value = reinterpret_cast<unsigned long long int*>(&value);
    auto* ptr_addr = reinterpret_cast<unsigned long long int*>(data_ + pos);

    auto result = atomicCAS(ptr_addr, *ptr_assumed, *ptr_value);
    return *reinterpret_cast<U*>(&result);
  }
};

// Check if type is a device array.
template<typename>
struct is_device_array : std::false_type {};

template<typename T, size_t N>
struct is_device_array<DeviceArray<T, N>> : std::true_type {};

#endif  // UTIL_UTIL_H
