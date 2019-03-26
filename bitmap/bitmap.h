#ifndef BITMAP_BITMAP_H
#define BITMAP_BITMAP_H

#include <assert.h>
#include <limits>
#include <stdint.h>
#include <cub/cub.cuh>

#include "bitmap/util.h"

#ifndef NDEBUG
static const int kMaxRetry = 10000000;
#define CONTINUE_RETRY (_retry_counter++ < kMaxRetry)
#define INIT_RETRY int _retry_counter = 0;
#define ASSERT_SUCCESS(expr) assert(expr);
#else
#define CONTINUE_RETRY (true)
#define INIT_RETRY
#define ASSERT_SUCCESS(expr) expr;
#endif  // NDEBUG

// Save memory: Allocate only one set of buffers.
// TODO: This should be stored in SoaAllocator class.
#define CUB_MAX_NUM_BLOCKS 64*64*64*64
__device__ int cub_enumeration_base_buffer[CUB_MAX_NUM_BLOCKS];
__device__ int cub_enumeration_id_buffer[CUB_MAX_NUM_BLOCKS];
__device__ int cub_enumeration_cub_temp[3*CUB_MAX_NUM_BLOCKS];
__device__ int cub_enumeration_cub_output[CUB_MAX_NUM_BLOCKS];
int* cub_enumeration_base_buffer_addr;
int* cub_enumeration_cub_temp_addr;
int* cub_enumeration_cub_output_addr;

void load_cub_buffer_addresses() {
  gpuErrchk(cudaGetSymbolAddress((void**) &cub_enumeration_base_buffer_addr,
                                 cub_enumeration_base_buffer));
  gpuErrchk(cudaGetSymbolAddress((void**) &cub_enumeration_cub_temp_addr,
                                 cub_enumeration_cub_temp));
  gpuErrchk(cudaGetSymbolAddress((void**) &cub_enumeration_cub_output_addr,
                                 cub_enumeration_cub_output));
}

void perform_cub_scan(int cub_scan_size) {
  size_t temp_size = 3*CUB_MAX_NUM_BLOCKS;
  gpuErrchk(cub::DeviceScan::InclusiveSum(cub_enumeration_cub_temp_addr,
                                          temp_size,
                                          cub_enumeration_base_buffer_addr,
                                          cub_enumeration_cub_output_addr,
                                          cub_scan_size));
}

static const int kCubScan = 0;
static const int kAtomicScan = 1;

static const int kNumBlocksAtomicAdd = 256;

template<typename SizeT, int NumContainers, int Bitsize>
struct ScanData {
  // Buffers for parallel enumeration.
  SizeT enumeration_result_buffer[NumContainers*Bitsize];
  SizeT enumeration_result_size;
};

// TODO: Only works with ContainerT = unsigned long long int.
template<typename SizeT, SizeT N, typename ContainerT = unsigned long long int,
         int ScanType = kAtomicScan>
class Bitmap {
 public:
  static const SizeT kIndexError = std::numeric_limits<SizeT>::max();
  static const ContainerT kZero = 0;
  static const ContainerT kOne = 1;

  struct BitPosition {
    __DEV__ BitPosition(SizeT index) : container_index(index / kBitsize),
                                       offset(index % kBitsize) {
      assert(index != kIndexError);
      assert(index < N);
    }

    SizeT container_index;
    SizeT offset;
  };

  Bitmap() = default;

  // Delete copy constructor.
  __DEV__ Bitmap(const Bitmap&) = delete;

  // Allocate specific index, i.e., set bit to 1. Return value indicates
  // success. If Retry, then continue retrying until successful update.
  template<bool Retry = false>
  __DEV__ bool allocate(SizeT index) {
    BitPosition pos(index);

    // Set bit to one.
    ContainerT pos_mask = kOne << pos.offset;
    ContainerT previous;
    bool success;

    INIT_RETRY;

    do {
      previous = atomicOr(data_.containers + pos.container_index, pos_mask);
      success = (previous & pos_mask) == 0;
    } while (Retry && !success && CONTINUE_RETRY);    // Retry until success

    if (kHasNested && success && previous == 0) {
      // Allocated first bit, propagate to nested.
      ASSERT_SUCCESS(data_.nested_allocate<true>(pos.container_index));
    }

    return success;
  }

  // Return the index of an allocated bit, utilizing the hierarchical bitmap
  // structure.
  template<bool Retry = false>
  __DEV__ SizeT find_allocated(int seed) const {
    SizeT index;

    INIT_RETRY;

    do {
      index = find_allocated_private(seed);
    } while (Retry && index == kIndexError && CONTINUE_RETRY);

    assert(!Retry || index != kIndexError);
    assert(index == kIndexError || index < N);

    return index;
  }

  // Deallocate arbitrary bit and return its index. Assuming that there is
  // at least one remaining allocation. Retries until success.
  __DEV__ SizeT deallocate() {
    SizeT index;
    bool success;

    INIT_RETRY;

    int retries = 0;
    do {
      index = find_allocated<false>(retries++);
      if (index == kIndexError) {
        success = false;
      } else {
        success = deallocate<false>(index); // if false: other thread was faster
      }
    } while (!success && CONTINUE_RETRY);

    assert(success);

    return index;
  }

  __DEV__ SizeT deallocate_seed(int seed) {
    SizeT index;
    bool success;

    INIT_RETRY;

    int retries = 0;
    do {
      index = find_allocated<false>(retries++ + seed);
      if (index == kIndexError) {
        success = false;
      } else {
        success = deallocate<false>(index); // if false: other thread was faster
      }
    } while (!success && CONTINUE_RETRY);

    assert(success);

    return index;
  }

  // Deallocate specific index, i.e., set bit to 0. Return value indicates
  // success. If Retry, then continue retrying until successful update.
  template<bool Retry = false>
  __DEV__ bool deallocate(SizeT index) {
    BitPosition pos(index);

    // Set bit to zero.
    ContainerT pos_mask = kOne << pos.offset;
    ContainerT previous;
    bool success;

    INIT_RETRY;

    do {
      previous = atomicAnd(data_.containers + pos.container_index, ~pos_mask);
      success = (previous & pos_mask) != 0;
    } while (Retry && !success && CONTINUE_RETRY);    // Retry until success

    if (kHasNested && success && __popcll(previous) == 1) {
      // Deallocated only bit, propagate to nested.
      ASSERT_SUCCESS(data_.nested_deallocate<true>(pos.container_index));
    }

    return success;
  }

  // Return the index of an arbitrary, allocated bit. This algorithm forwards
  // the request to the top-most (deepest nested) bitmap and returns the index
  // of the container in which to search. On the lowest level, the container ID
  // equals the bit index.
  // TODO: Sould be private.
  __DEV__ SizeT find_allocated_private(int seed) const {
    SizeT container;

    if (kHasNested) {
      container = data_.nested_find_allocated_private(seed);

      if (container == kIndexError) {
        return kIndexError;
      }
    } else {
      container = 0;
    }

    return find_allocated_in_container(container, seed);
  }

  // Copy other bitmap.
  __DEV__ void initialize(const Bitmap<SizeT, N, ContainerT, ScanType>& other) {
    for (SizeT i = blockIdx.x*blockDim.x + threadIdx.x;
         i < kNumContainers;
         i += blockDim.x*gridDim.x) {
      data_.containers[i] = other.data_.containers[i];
    }

    if (kHasNested) {
      data_.nested_initialize(other.data_);
    }
  }
  // Initialize bitmap to all 0 or all 1.
  __DEV__ void initialize(bool allocated = false) {
    for (SizeT i = blockIdx.x*blockDim.x + threadIdx.x;
         i < kNumContainers;
         i += blockDim.x*gridDim.x) {
      if (allocated) {
        if (i == kNumContainers - 1 && N % kBitsize > 0) {
          // Last container is only partially occupied.
          data_.containers[i] = (kOne << (N % kBitsize)) - 1;
        } else if (i < kNumContainers) {
          data_.containers[i] = ~kZero;
        }
      } else {
        data_.containers[i] = static_cast<ContainerT>(0);
      }
    }

    if (kHasNested) {
      data_.nested_initialize(allocated);
    }
  }

  __DEV__ SizeT DBG_count_num_ones() {
    return data_.DBG_count_num_ones();
  }

  // Return true if index is allocated.
  __DEV__ bool operator[](SizeT index) const {
    return data_.containers[index/kBitsize] & (kOne << (index % kBitsize));
  }

  __DEV__ ContainerT get_container(SizeT index) const {
    return data_.containers[index];
  }

  // Initiate scan operation (from the host side). This request is forwarded
  // to the next-level bitmap. Afterwards, scan continues here.
  void scan() {
    gpuErrchk(cudaPeekAtLastError());
    data_.scan();
  }

  // May only be called after scan.
  __DEV__ SizeT scan_num_bits() const {
    return data_.scan_data.enumeration_result_size;
  }

  __host__ __device__ SizeT* scan_num_bits_ptr() {
    return &data_.scan_data.enumeration_result_size;
  }

  // Returns the index of the pos-th set bit.
  __DEV__ SizeT scan_get_index(SizeT pos) const {
    return data_.scan_data.enumeration_result_buffer[pos];
  }

  // Nested bitmap data structure.
  template<SizeT NumContainers, bool HasNested>
  struct BitmapData;

  // Bitmap data structure with a nested bitmap.
  template<SizeT NumContainers>
  struct BitmapData<NumContainers, true> {
    using ThisClass = BitmapData<NumContainers, true>;

    static const uint8_t kBitsize = 8*sizeof(ContainerT);

    ContainerT containers[NumContainers];

    // Buffers for parallel enumeration.
    ScanData<SizeT, NumContainers, kBitsize> scan_data;

    // Containers that store the bits.
    using BitmapT = Bitmap<SizeT, NumContainers, ContainerT, ScanType>;
    BitmapT nested;

    static const int kLevels = 1 + BitmapT::kLevels;

    __DEV__ SizeT DBG_count_num_ones() {
      SizeT result = 0;
      for (int i = 0; i < NumContainers; ++i) {
        result += __popcll(containers[i]);
      }
      return result;
    }

    // Allocate a specific bit in the nested bitmap.
    template<bool Retry>
    __DEV__ bool nested_allocate(SizeT pos) {
      return nested.allocate<Retry>(pos);
    }

    // Deallocate a specific bit in the nested bitmap.
    template<bool Retry>
    __DEV__ bool nested_deallocate(SizeT pos) {
      return nested.deallocate<Retry>(pos);
    }

    // Find an allocated bit in the nested bitmap.
    __DEV__ SizeT nested_find_allocated_private(int seed) const {
      return nested.find_allocated_private(seed);
    }

    __DEV__ void nested_initialize(
        const BitmapData<NumContainers, true>& other) {
      nested.initialize(other.nested);
    }

    // Initialize the nested bitmap.
    __DEV__ void nested_initialize(bool allocated) {
      nested.initialize(allocated);
    }

    template<int S = ScanType>
    __DEV__ typename std::enable_if<S == kCubScan, void>::type
    set_result_size() {
      SizeT num_selected = nested.data_.scan_data.enumeration_result_size;
      // Base buffer contains prefix sum.
      SizeT result_size = cub_enumeration_cub_output[
          num_selected*kBitsize - 1];
      scan_data.enumeration_result_size = result_size;
      assert(result_size >= num_selected);
    }

    __DEV__ void set_result_size_to_zero() {
      scan_data.enumeration_result_size = 0;
    }

    // TODO: Run with num_selected threads, then we can remove the loop.
    template<int S = ScanType>
    __DEV__ typename std::enable_if<S == kCubScan, void>::type
    pre_scan() {
      SizeT* selected = nested.data_.scan_data.enumeration_result_buffer;
      SizeT num_selected = nested.data_.scan_data.enumeration_result_size;

      for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
           sid < num_selected; sid += blockDim.x * gridDim.x) {
        SizeT container_id = selected[sid];
        auto value = containers[container_id];
        for (int i = 0; i < kBitsize; ++i) {
          // Write "1" if allocated, "0" otherwise.
          bool bit_selected = value & (static_cast<ContainerT>(1) << i);
          cub_enumeration_base_buffer[sid*kBitsize + i] = bit_selected;
          if (bit_selected) {
            cub_enumeration_id_buffer[sid*kBitsize + i] =
                kBitsize*container_id + i;
          }
        }
      }
    }

    // Run with num_selected threads.
    // Assumption: enumeration_base_buffer contains exclusive prefix sum.
    template<int S = ScanType>
    __DEV__ typename std::enable_if<S == kCubScan, void>::type
    post_scan() {
      SizeT* selected = nested.data_.scan_data.enumeration_result_buffer;
      SizeT num_selected = nested.data_.scan_data.enumeration_result_size;

      for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
           sid < num_selected; sid += blockDim.x * gridDim.x) {
        SizeT container_id = selected[sid];
        auto value = containers[container_id];
        for (int i = 0; i < kBitsize; ++i) {
          // Write "1" if allocated, "0" otherwise.
          bool bit_selected = value & (static_cast<ContainerT>(1) << i);
          if (bit_selected) {
            // Minus one because scan operation is inclusive.
            scan_data.enumeration_result_buffer[
                cub_enumeration_cub_output[sid*kBitsize + i] - 1] =
                    cub_enumeration_id_buffer[sid*kBitsize + i];
          }
        }
      }
    }

    template<int S = ScanType>
    typename std::enable_if<S == kCubScan, void>::type
    scan() {  
      nested.scan();

      SizeT num_selected = read_from_device<SizeT>(
          &nested.data_.scan_data.enumeration_result_size);

      if (num_selected > 0) {
        member_func_kernel<ThisClass, &ThisClass::pre_scan>
            <<<num_selected/256+1, 256>>>(this);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Workaround due to bug in CUB.
        perform_cub_scan(num_selected*kBitsize);

        member_func_kernel<ThisClass, &ThisClass::post_scan>
            <<<num_selected/256+1, 256>>>(this);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        member_func_kernel<ThisClass, &ThisClass::set_result_size><<<1, 1>>>(this);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
      } else {
        member_func_kernel<ThisClass, &ThisClass::set_result_size_to_zero>
            <<<1, 1>>>(this);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
      }
    }

    template<int S = ScanType>
    __DEV__ typename std::enable_if<S == kAtomicScan, void>::type
    atomic_add_scan_init() {
      scan_data.enumeration_result_size = 0;
    }

    template<int S = ScanType>
    __DEV__ typename std::enable_if<S == kAtomicScan, void>::type
    atomic_add_scan() {
      SizeT* selected = nested.data_.scan_data.enumeration_result_buffer;
      SizeT num_selected = nested.data_.scan_data.enumeration_result_size;
      //printf("num_selected=%i\n", (int) num_selected);

      for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
           sid < num_selected; sid += blockDim.x * gridDim.x) {
        SizeT container_id = selected[sid];
        auto value = containers[container_id];
        int num_bits = __popcll(value);

        auto before = atomicAdd(
            reinterpret_cast<unsigned int*>(&scan_data.enumeration_result_size),
            num_bits);

        for (int i = 0; i < num_bits; ++i) {
          int next_bit = __ffsll(value) - 1;
          assert(next_bit >= 0);
          scan_data.enumeration_result_buffer[before + i] =
              container_id*kBitsize + next_bit;

          // Advance to next bit.
          value &= value - 1;
        }
      }      
    }

    template<int S = ScanType>
    typename std::enable_if<S == kAtomicScan, void>::type scan() {
      nested.scan();

      SizeT num_selected = read_from_device<SizeT>(
          &nested.data_.scan_data.enumeration_result_size);

      if (num_selected > 0) {
        member_func_kernel<ThisClass, &ThisClass::atomic_add_scan_init>
            <<<1, 1>>>(this);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        member_func_kernel<ThisClass, &ThisClass::atomic_add_scan>
            <<<(num_selected + kNumBlocksAtomicAdd - 1)/kNumBlocksAtomicAdd,
               kNumBlocksAtomicAdd>>>(this);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
      } else {
        member_func_kernel<ThisClass, &ThisClass::set_result_size_to_zero>
            <<<1, 1>>>(this);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
      }
    }
  };

  // Bitmap data structure without a nested bitmap.
  template<SizeT NumContainers>
  struct BitmapData<NumContainers, false> {
    static_assert(NumContainers == 1,
                  "L0 bitmap should have only one container.");

    using ThisClass = BitmapData<NumContainers, false>;

    static const uint8_t kBitsize = 8*sizeof(ContainerT);

    static const int kLevels = 1;

    // Bitmaps without a nested bitmap have exactly one container.
    ContainerT containers[NumContainers];

    // Buffers for parallel enumeration.
    ScanData<SizeT, NumContainers, kBitsize> scan_data;

    __DEV__ SizeT DBG_count_num_ones() {
      return __popcll(containers[0]);
    }

    template<bool Retry>
    __DEV__ bool nested_allocate(SizeT pos) { assert(false); return false; }

    template<bool Retry>
    __DEV__ bool nested_deallocate(SizeT pos) { assert(false); return false; }

    __DEV__ SizeT nested_find_allocated_private(int seed) const {
      assert(false);
      return kIndexError;
    }

    __DEV__ void nested_initialize(
        const BitmapData<NumContainers, false>& other) { assert(false); }

    __DEV__ void nested_initialize(bool allocated) { assert(false); }

    __DEV__ void trivial_scan() {
      assert(blockDim.x == 64 && gridDim.x == 1);
      auto val = containers[0];

      if (val & (kOne << threadIdx.x)) {
        // Count number of bits before threadIdx.x.
        int pos = __popcll(val & ((1ULL << threadIdx.x) - 1));
        scan_data.enumeration_result_buffer[pos] = threadIdx.x;
      }

      scan_data.enumeration_result_size = __popcll(val);
    }

    void scan() {
      // Does not perform a prefix scan but computes the result directly.
      member_func_kernel<ThisClass, &ThisClass::trivial_scan>
          <<<1, 64>>>(this);
      gpuErrchk(cudaDeviceSynchronize());
    }
  };

  // Returns the index of an allocated bit inside a container. Returns
  // kIndexError if not allocated bit was found.
  __DEV__ SizeT find_allocated_in_container(SizeT container, int seed) const {
    // TODO: For better performance, choose random one.
    int selected = find_allocated_bit(data_.containers[container], seed);
    if (selected == -1) {
      // No space in here.
      return kIndexError;
    } else {
      return selected + container*kBitsize;
    }
  }

  __DEV__ int find_first_bit(ContainerT val, int seed) const {
    // TODO: Adapt for other data types.
    return __ffsll(val) - 1;
  }

  __DEV__ int find_allocated_bit(ContainerT val, int seed) const {
    return find_allocated_bit_fast(val, seed);
  }

  // Find index of *some* bit that is set to 1.
  // TODO: Make this more efficient!
  __DEV__ int find_allocated_bit_fast(ContainerT val, int seed) const {
    unsigned int rotation_len = (seed+warp_id()) % (sizeof(val)*8);
    const ContainerT rotated_val = rotl(val, rotation_len);

    int first_bit_pos = __ffsll(rotated_val) - 1;
    if (first_bit_pos == -1) {
      return -1;
    } else {
      return (first_bit_pos - rotation_len) % (sizeof(val)*8);
    }
  }

  __DEV__ int find_allocated_bit_compact(ContainerT val) const {
    return __ffsll(val) - 1;
  }

  template<typename F, typename... Args>
  __DEV__ void enumerate(F func, Args... args);

  // The number of bits per container.
  static const uint8_t kBitsize = 8*sizeof(ContainerT);

  // The number of containers on this level.
  static const SizeT kNumContainers = N <= kBitsize ? 1 : N / kBitsize;

  // Indicates if this bitmap has a higher-level (nested) bitmap.
  static const bool kHasNested = kNumContainers > 1;

  static_assert(!kHasNested || N % kBitsize == 0,
                "N must be of size (sizeof(ContainerT)*8)**D * x.");

  using SizeTT = SizeT;
  using ContainerTT = ContainerT;

  // Nested bitmap structure.
  using BitmapDataT = BitmapData<kNumContainers, kHasNested>;
  BitmapDataT data_;

  // Number of bitmap levels, including this one.
  static const int kLevels = BitmapDataT::kLevels;

  // Type of outer bitmap.
  using OuterBitmapT = Bitmap<SizeT, N*kBitsize, ContainerT, ScanType>;
};

#include "bitmap/sequential_enumerate.h"

#endif  // BITMAP_BITMAP_H
