#ifndef UTIL_UTIL_H
#define UTIL_UTIL_H

#include <type_traits>

#define __DEV__ __device__
#define __DEV_HOST__ __device__ __host__

#ifdef __CUDA_ARCH__
#define __host_or_device__ __device__
#else
#define __host_or_device__
#endif  // __CUDA_ARCH__

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
  static const int kN = N;

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
__forceinline__ static T read_from_device(T* ptr) {
  T host_storage;
  cudaMemcpy(&host_storage, ptr, sizeof(T), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaDeviceSynchronize());
  return host_storage;
}

// A wrapper that runs a device member function.
template<typename C, void (C::*func)()>
__global__ static void member_func_kernel(C* ptr) {
  (ptr->*func)();
}

template<typename C, typename R, R (C::*func)()>
__global__ static void member_func_kernel_return(C* ptr, R* result) {
  *result = (ptr->*func)();
}

template<typename C, typename R, R (C::*func)()>
R call_return_member_func(C* ptr) {
  R h_result;
  R* d_result;
  cudaMalloc(&d_result, sizeof(R));

  member_func_kernel_return<C, R, func><<<1, 1>>>(ptr, d_result);
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(&h_result, d_result, sizeof(R), cudaMemcpyDeviceToHost);
  cudaFree(d_result);

  return h_result;
}

template<typename C, typename T1, void (C::*func)(T1)>
__global__ static void member_func_kernel(C* ptr, T1 t1) {
  (ptr->*func)(t1);
}

template<typename C, typename T1, typename T2, void (C::*func)(T1, T2)>
__global__ static void member_func_kernel(C* ptr, T1 t1, T2 t2) {
  (ptr->*func)(t1, t2);
}

#ifdef __CUDA_ARCH__
__forceinline__ __host__ __device__ int bit_ffsll(long long int value) {
  return __ffsll(value);
}

__forceinline__ __host__ __device__ int bit_ffsll(unsigned long long int value) {
  return __ffsll(*reinterpret_cast<long long int*>(&value));
}
#else
__forceinline__ __host__ __device__ int bit_ffsll(unsigned long long int value) {
  return __builtin_ffsll(value);
}
#endif  // __CUDA_ARCH__

__forceinline__ __host__ __device__ int bit_popcll(unsigned long long int value) {
#ifdef __CUDA_ARCH__
  return __popcll(value);
#else
  return __builtin_popcountll(value);
#endif  // __CUDA_ARCH__
}

// Shift left, rotating.
// Copied from: https://gist.github.com/pabigot/7550454
template <typename T>
__device__ __host__ T rotl (T v, unsigned int b)
{
  static_assert(std::is_integral<T>::value, "rotate of non-integral type");
  static_assert(! std::is_signed<T>::value, "rotate of signed type");
  constexpr unsigned int num_bits {std::numeric_limits<T>::digits};
  static_assert(0 == (num_bits & (num_bits - 1)), "rotate value bit length not power of two");
  constexpr unsigned int count_mask {num_bits - 1};
  const unsigned int mb {b & count_mask};
  using promoted_type = typename std::common_type<int, T>::type;
  using unsigned_promoted_type = typename std::make_unsigned<promoted_type>::type;
  return ((unsigned_promoted_type{v} << mb)
          | (unsigned_promoted_type{v} >> (-mb & count_mask)));
}

// Seems like this is a scheduler warp ID and may change.
__forceinline__ __device__ unsigned warp_id()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

__device__ __inline__ unsigned long long int ld_gbl_cg(
    const unsigned long long int *addr) {
  unsigned long long int return_value;
  asm("ld.global.cg.s64 %0, [%1];" : "=l"(return_value) : "l"(addr));
  return return_value;
}

#endif  // UTIL_UTIL_H
