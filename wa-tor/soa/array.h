#ifndef WA_TOR_SOA_ARRAY_H
#define WA_TOR_SOA_ARRAY_H

// Custom std::array because std::array is not available on device.
template<typename T, size_t N>
class DevArray {
 private:
  T data[N];

 public:
  __device__ T& operator[](size_t pos) {
    return data[pos];
  }
};

#endif  // WA_TOR_SOA_ARRAY_H
