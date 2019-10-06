#ifndef BITMAP_BITMAP_H
#define BITMAP_BITMAP_H

#include <assert.h>
#include <limits>
#include <stdint.h>
#include <cub/cub.cuh>

#include "util/util.h"

#ifndef NDEBUG
/**
 * The maximum number of times that a bitmap update is retried. (If it fails
 * due to bitmap inconsistencies.)
 */
static const int kMaxRetry = 10000000;

/**
 * Condition for doing another iteration.
 */
#define CONTINUE_RETRY (_retry_counter++ < kMaxRetry)

/**
 * Initializes the retry counter.
 */
#define INIT_RETRY int _retry_counter = 0;

/**
 * In debug mode, \p expr must evaluate to true.
 */
#define ASSERT_SUCCESS(expr) assert(expr);

#else

/**
 * Number of bitmap update retries is not limited in optimized mode.
 */
#define CONTINUE_RETRY (true)

/**
 * No retry counter needed in optimized mode.
 */
#define INIT_RETRY
/**
 * \p expr should evaluate to true, but in optimized mode this is not enforced.
 */
#define ASSERT_SUCCESS(expr) expr;
#endif  // NDEBUG

// TODO: Save memory: Allocate only one set of buffers.
// TODO: This should be stored in SoaAllocator class.
// TODO: Needs refactoring.
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

/**
 * A bitmap consisting of \p N bits. To accelerate certain bitmap operations,
 * this class maintains a hierarchy of bitmaps. This hierarchy is hidden from
 * the bitmap's public API.
 * @tparam SizeT Data type of indices. Usually int or unsigned int.
 * @tparam N Number of bits.
 * @tparam ContainerT Data type that is used internally to store bits. This
 *                    is currently fixed to unsigned long long int.
 * @tparam ScanType Scan strategy. Scanning = enumerating all set bits. There
 *                  are two possible strategies: Prefix sum-based CUB scan and
 *                  scan based on atomic operations.
 */
template<typename SizeT, SizeT N, typename ContainerT = unsigned long long int,
         int ScanType = kAtomicScan>
class Bitmap {
 public:
  /**
   * Error code. This value is returned if a set bit was requested but none was
   * found, e.g., because the bitmaps has no set bits.
   */
  static const SizeT kIndexError = std::numeric_limits<SizeT>::max();

  /**
   * Shortcut for constant 0.
   */
  static const ContainerT kZero = 0;

  /**
   * Shortcut for constant 1.
   */
  static const ContainerT kOne = 1;

  /**
   * Represents the position of a bit in this bitmap.
   */
  struct BitPosition {
    __device__ __host__ BitPosition(SizeT index)
        : container_index(index / kBitsize), offset(index % kBitsize) {
      assert(index != kIndexError);
      assert(index < N);
    }

    /**
     * Index of container.
     */
    SizeT container_index;

    /**
     * Index of bit within the container.
     */
    SizeT offset;
  };

  /**
   * Note: Bitmaps should be initialized with Bitmap::initialize().
   */
  Bitmap() = default;

  /**
   * Initializes this bitmap by copying another bitmap.
   * @param other Bitmap to copy.
   */
  __device__ __host__ Bitmap(const Bitmap& other) { initialize(other); }

  /**
   * Initializes this bitmap with all zeros or all ones.
   * @param state false indicates all zeros, true indicates all ones.
   */
  __device__ __host__ Bitmap(bool state) { initialize(state); }

  /**
   * Initializes this bitmap by copying another bitmap.
   * @param other Source bitmap
   */
  __device__ __host__ void initialize(
      const Bitmap<SizeT, N, ContainerT, ScanType>& other) {
#ifdef __CUDA_ARCH__
    for (SizeT i = blockIdx.x * blockDim.x + threadIdx.x;
         i < kNumContainers;
         i += blockDim.x * gridDim.x) {
#else
    for (SizeT i = 0; i < kNumContainers; ++i) {
#endif  // __CUDA_ARCH__
      data_.containers[i] = other.data_.containers[i];
    }

    if (kHasNested) { data_.nested_initialize(other.data_); }
  }

  /**
   * Initializes this bitmap with all 0 or all 1.
   * @param allocated true indicates all 1, falses indicates all 0
   */
  __device__ __host__ void initialize(bool allocated = false) {
#ifdef __CUDA_ARCH__
    for (SizeT i = blockIdx.x * blockDim.x + threadIdx.x;
         i < kNumContainers;
         i += blockDim.x * gridDim.x) {
#else
    for (SizeT i = 0; i < kNumContainers; ++i) {
#endif  // __CUDA_ARCH_
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

    if (kHasNested) { data_.nested_initialize(allocated); }
  }

  /**
   * Allocates, i.e., sets a bit to 1, at a specific index. The return value
   * indicates if this operation switched the bit to 1. If the bit was already
   * 1, then this function returns false. If \p Retry is set to true, then this
   * function keeps trying to set the bit to 1 until the bit was actually
   * changed from 0 to 1 by this function call.
   * * If \p Retry is false, then this function corresponds to try_set() in the
   *   paper.
   * * If \p Retry is true, then this function corresponds to set() the paper.
   * @tparam Retry Indicates whether the operation should retry until the bit
   *         was actually switched/modified.
   * @param index Index of the bit.
   */
  template<bool Retry = false>
  __device__ bool allocate(SizeT index) {
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

  /**
   * Finds and returns the index of an allocated (set) bit. This function uses
   * the internal bitmap hierarchy for better performance. If \p Retry is
   * false, then this function corresponds to try_find_set() in the paper. If
   * \p Retry is true, then this function retries until a bit was found. This
   * may result in an endless loop.
   * @tparam Retry Indicates whether the operation should retry until a set bit
   *         was located.
   * @param seed A seed that is used to randomize the hierarchy/tree traversal.
   */
  template<bool Retry = false>
  __device__ __host__ SizeT find_allocated(int seed) const {
    SizeT index;

    INIT_RETRY;

    // TODO: Modify seed with every retry?
    do {
      index = try_find_allocated(seed);
    } while (Retry && index == kIndexError && CONTINUE_RETRY);

    assert(!Retry || index != kIndexError);
    assert(index == kIndexError || index < N);

    return index;
  }

  /**
   * Deallocates (clears) an arbitrary bit and returns its index, assuming that
   * there is at least one remaining set bit. Retries until a bit was found and
   * cleared. Corresponds to clear() in the paper.
   */
  __device__ SizeT deallocate() {
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

  /**
   * Same as Bitmap::deallocate(), but a specific seed can be specified.
   */
  __device__ SizeT deallocate_seed(int seed) {
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

  /**
   * Deallocates (clears) a bit at a specific index, i.e., sets the bit to 0.
   * If \p Retry is true, then this function repeatedly attempts to flip the
   * bit to 0 and only sbtops if this function invocation actually changed
   * the bit. In that case, the return value of the function is always true.
   * Otherwise, the return value of the function indicates whether the bit was
   * successfully flipped. (false return values indicate that the bit was
   * already 0.)
   * * If \p Retry is false, then this function corresponds to try_clear()
   *   in the paper.
   * * If \p Retry is true, then this function corresponds to clear() in the
   *   paper. (The variant that takes an argument.)
   * @tparam Retry Indicates whether the operation should be repeated until bit
   *         was actually changed.
   * @param index Position/index of the bit.
   */
  template<bool Retry = false>
  __device__ bool deallocate(SizeT index) {
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

    if (kHasNested && success && bit_popcll(previous) == 1) {
      // Deallocated only bit, propagate to nested.
      ASSERT_SUCCESS(data_.nested_deallocate<true>(pos.container_index));
    }

    return success;
  }

  /**
   * Returns the index of an arbitrary set bit. This algorithm forwards the
   * request to the higher-level (nested) bitmap, which returns the index
   * of the container in which to search. On the lowest level (usually the
   * level of the initial function invocation), the container ID equals the
   * index of the selected bit.
   * @param seed Seed value for randomizing the hierarchy traversal.
   * TODO: Not public API, function should be private.
   */
  __device__ __host__ SizeT try_find_allocated(int seed) const {
    SizeT container;

    if (kHasNested) {
      container = data_.nested_try_find_allocated(seed);

      if (container == kIndexError) {
        return kIndexError;
      }
    } else {
      container = 0;
    }

    return find_allocated_in_container(container, seed);
  }

  /**
   * Only for debugging: Counts the number of set bits.
   */
  __device__ __host__ SizeT DBG_count_num_ones() {
    return data_.DBG_count_num_ones();
  }

  /**
   * Checks if the index-th bit is set.
   * @param index Position of bit
   */
  __device__ __host__ bool operator[](SizeT index) const {
    return data_.containers[index/kBitsize] & (kOne << (index % kBitsize));
  }

  /**
   * Returns a container of bits.
   * @param index Index of container
   */
  __device__ __host__ ContainerT get_container(SizeT index) const {
    return data_.containers[index];
  }

  /**
   * Initiates a bitmap scan operation. The \p ScanType class template
   * parameter determines which strategy is used. Afterwards, scan_num_bits(),
   * scan_num_bits_ptr() and scan_get_index() can be invoked.
   * Note: This function is part of the indices() operation in the paper.
   * Note: This function must be invoked from the host side. Internally, it
   * launches multiple CUDA kernels.
   */
  void scan() {
    gpuErrchk(cudaPeekAtLastError());
    data_.scan();
  }

  /**
   * Returns the number of set bits in the bitmap. Can only be called after a
   * scan().
   */
  __DEV__ SizeT scan_num_bits() const {
    return data_.scan_data.enumeration_result_size;
  }

  /**
   * Returns a pointer to a memory address that contains the number of set bits
   * in the bitmap. Can only be called after a scan().
   */
  __host__ __device__ SizeT* scan_num_bits_ptr() {
    return &data_.scan_data.enumeration_result_size;
  }

  /**
   * Returns the index of the pos-th set bit in the bitmap. Can only be called
   * after a scan().
   */
  __device__ __host__ SizeT scan_get_index(SizeT pos) const {
    return data_.scan_data.enumeration_result_buffer[pos];
  }

  /**
   * Contains the bitmap itself and maybe a nested bitmap. There are template
   * specialization with and without a nested bitmap.
   * @tparam NumContainers The number of containers (size of the bitmap).
   * @tparam HasNested Indicates whether this bitmap has a nested bitmap.
   */
  template<SizeT NumContainers, bool HasNested>
  struct BitmapData;

  /**
   * BitmapData specialization that contains a nested bitmap.
   * @tparam NumContainers The number of containers (size of the bitmap).
   */
  template<SizeT NumContainers>
  struct BitmapData<NumContainers, true> {
    /**
     * A shortcut for this template instantiation.
     */
    using ThisClass = BitmapData<NumContainers, true>;

    /**
     * The number of bits per container.
     */
    static const uint8_t kBitsize = 8*sizeof(ContainerT);

    /**
     * An array of containers. This is the actual bitmap.
     */
    ContainerT containers[NumContainers];

    /**
     * Stores the result of a scan operation.
     */
    ScanData<SizeT, NumContainers, kBitsize> scan_data;

    /**
     * Type of the nested bitmap. The number of containers of this bitmap is
     * the number of bits of the nested bitmap.
     */
    using BitmapT = Bitmap<SizeT, NumContainers, ContainerT, ScanType>;

    /**
     * The nested (higher-level) bitmap.
     */
    BitmapT nested;

    /**
     * The level of this bitmap. Contrary to the the paper, the innermost
     * bitmap has level 1 and we keep counting with every outer bitmap.
     */
    static const int kLevels = 1 + BitmapT::kLevels;

    /**
     * Only for debugging: Counts the number of ones.
     */
    __device__ __host__ SizeT DBG_count_num_ones() {
      SizeT result = 0;
      for (int i = 0; i < NumContainers; ++i) {
        result += bit_popcll(containers[i]);
      }
      return result;
    }

    /**
     * Allocates a specific bit in BitmapData::nested.
     * @tparam Retry Indicates whether the operation retries until the bit was
     *               actually modified.
     */
    template<bool Retry>
    __device__ bool nested_allocate(SizeT pos) {
      return nested.allocate<Retry>(pos);
    }

    /**
     * Clears a specific bit in BitmapData::nested.
     * @tparam Retry Indicates whether the operation retries until the bit was
     *               actually modified.
     */
    template<bool Retry>
    __device__ bool nested_deallocate(SizeT pos) {
      return nested.deallocate<Retry>(pos);
    }

    /**
     * Find and return the position of a set bit in BitmapData::nested.
     * @param seed Seed for randomizing the tree traversal.
     */
    __device__ __host__ SizeT nested_try_find_allocated(int seed) const {
      return nested.try_find_allocated(seed);
    }

    /**
     * Initializes the bitmap from another bitmap.
     * @param other Bitmap to be copied.
     */
    __device__ __host__ void nested_initialize(
        const BitmapData<NumContainers, true>& other) {
      nested.initialize(other.nested);
    }

    /**
     * Initializes the bitmap with all zeros or all ones.
     * @param allocated Zeros or ones.
     */
    __device__ __host__ void nested_initialize(bool allocated) {
      nested.initialize(allocated);
    }

    template<int S = ScanType>
    __device__ typename std::enable_if<S == kCubScan, void>::type
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
    __device__ typename std::enable_if<S == kCubScan, void>::type
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
    __device__ typename std::enable_if<S == kCubScan, void>::type
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

    /**
     * A prefix sum-based (CUB) scan works as follows. Process is described in
     * detail in the paper.
     * 1. Run scan operation on nested bitmap.
     * 2. Get number of ones in nested bitmap. If none, exit.
     * 3. Pre-Scan: Generate an integer buffer of size "number of bits". If a
     *    bit is set, then the corresponding buffer element is 1, otherwise 0.
     * 4. Run CUB scan (prefix sum).
     * 5. Post-Scan: For every set bit, write the corresponding prefix sum
     *    result value to the result array at the same location.
     * @tparam S Required for enable_if.
     */
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
    __device__ typename std::enable_if<S == kAtomicScan, void>::type
    atomic_add_scan_init() {
      scan_data.enumeration_result_size = 0;
    }

    template<int S = ScanType>
    __device__ typename std::enable_if<S == kAtomicScan, void>::type
    atomic_add_scan() {
      SizeT* selected = nested.data_.scan_data.enumeration_result_buffer;
      SizeT num_selected = nested.data_.scan_data.enumeration_result_size;
      //printf("num_selected=%i\n", (int) num_selected);

      for (int sid = threadIdx.x + blockIdx.x * blockDim.x;
           sid < num_selected; sid += blockDim.x * gridDim.x) {
        SizeT container_id = selected[sid];
        auto value = containers[container_id];
        int num_bits = bit_popcll(value);

        auto before = atomicAdd(
            reinterpret_cast<unsigned int*>(&scan_data.enumeration_result_size),
            num_bits);

        for (int i = 0; i < num_bits; ++i) {
          int next_bit = bit_ffsll(value) - 1;
          assert(next_bit >= 0);
          scan_data.enumeration_result_buffer[before + i] =
              container_id*kBitsize + next_bit;

          // Advance to next bit.
          value &= value - 1;
        }
      }      
    }

    /**
     * An atomic scan works as follows. Process is described in detail in the
     * paper (indicies() operation).
     * 1. Run scan operation on nested bitmap.
     * 2. Get number of ones in nested bitmap. If none, exit.
     * 3. Init: Set result size to 0.
     * 4. Atomic Scan: For every set bit, increase an atomic counter. As an
     *    optimization, this is done on a per-container basis.
     * @tparam S Required for enable_if.
     */
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

  /**
   * BitmapData specialization that does not contain a nested bitmap.
   * @tparam NumContainers The number of containers (size of the bitmap).
   */
  template<SizeT NumContainers>
  struct BitmapData<NumContainers, false> {
    static_assert(NumContainers == 1,
                  "L0 bitmap should have only one container.");

    /**
     * A shortcut for this template instantiation.
     */
    using ThisClass = BitmapData<NumContainers, false>;

    /**
     * The number of bits per container.
     */
    static const uint8_t kBitsize = 8*sizeof(ContainerT);

    /**
     * The top-level bitmap has exactly one level.
     */
    static const int kLevels = 1;

    /**
     * Bitmaps without a nested bitmap have exactly one container.
     */
    ContainerT containers[NumContainers];

    /**
     * Stores the result of a scan operation.
     */
    ScanData<SizeT, NumContainers, kBitsize> scan_data;

    /**
     * Only for debugging: Counts the number of ones.
     */
    __device__ __host__ SizeT DBG_count_num_ones() {
      return bit_popcll(containers[0]);
    }

    /**
     * Function is never called, but required for typing reasons.
     */
    template<bool Retry>
    __device__ bool nested_allocate(SizeT pos) { assert(false); return false; }

    /**
     * Function is never called, but required for typing reasons.
     */
    template<bool Retry>
    __device__ bool nested_deallocate(SizeT pos) {
      assert(false); return false;
    }

    /**
     * Function is never called, but required for typing reasons.
     */
    __device__ __host__ SizeT nested_try_find_allocated(int seed) const {
      assert(false);
      return kIndexError;
    }

    /**
     * Function is never called, but required for typing reasons.
     */
    __device__ __host__ void nested_initialize(
        const BitmapData<NumContainers, false>& other) { assert(false); }

    /**
     * Function is never called, but required for typing reasons.
     */
    __device__ __host__ void nested_initialize(bool allocated) {
      assert(false);
    }

    /**
     * Counts the number and indicies of set bits with population count.
     */
    __device__ void trivial_scan() {
      assert(blockDim.x == 64 && gridDim.x == 1);
      auto val = containers[0];

      if (val & (kOne << threadIdx.x)) {
        // Count number of bits before threadIdx.x.
        int pos = bit_popcll(val & ((1ULL << threadIdx.x) - 1));
        scan_data.enumeration_result_buffer[pos] = threadIdx.x;
      }

      scan_data.enumeration_result_size = bit_popcll(val);
    }

    /**
     * This bitmap has only one container, so scan is trivial.
     */
    void scan() {
      // Does not perform a prefix scan but computes the result directly.
      member_func_kernel<ThisClass, &ThisClass::trivial_scan>
          <<<1, 64>>>(this);
      gpuErrchk(cudaDeviceSynchronize());
    }
  };

  /**
   * Returns the index of an allocated bit inside a container. Returns
   * kIndexError if no allocated bit was found.
   * @param container Container of bits.
   * @param seed Seed for randomizing traversals.
   */
  __device__ __host__ SizeT find_allocated_in_container(
      SizeT container, int seed) const {
    int selected = find_allocated_bit(data_.containers[container], seed);
    if (selected == -1) {
      // No space in here.
      return kIndexError;
    } else {
      return selected + container * kBitsize;
    }
  }

  /**
   * This is an alternative strategy to find_allocated_bit_fast(). This
   * strategy always selects the first set bit. Results in poor performance.
   * @param val Container of bits.
   * @param seed Not utilized.
   */
  __device__ __host__ int find_first_bit(ContainerT val, int seed) const {
    // TODO: Adapt for other data types.
    return bit_ffsll(val) - 1;
  }

  __device__ __host__ int find_allocated_bit(ContainerT val, int seed) const {
    return find_allocated_bit_fast(val, seed);
  }

  /**
   * Find the position of a bit that is set to 1 in a given container. To
   * avoid multiple threads from selecting the same bit, the bitmap is first
   * rotation-shifted by \p seed and the warp ID.
   * @param val Container of bits.
   * @param seed Seed for randomizing traversals.
   */
  __device__ __host__ int find_allocated_bit_fast(ContainerT val, int seed) const {
#ifdef __CUDA_ARCH__
    // TODO: Can this be more efficient?
    unsigned int rotation_len = (seed+warp_id()) % (sizeof(val)*8);
#else
    unsigned int rotation_len = (seed+rand()) % (sizeof(val)*8);
#endif  // __CUDA_ARCH__

    const ContainerT rotated_val = rotl(val, rotation_len);

    int first_bit_pos = bit_ffsll(rotated_val) - 1;
    if (first_bit_pos == -1) {
      return -1;
    } else {
      // Shift result back.
      return (first_bit_pos - rotation_len) % (sizeof(val)*8);
    }
  }

  /**
   * Enumerate all set bits sequentially. This function is required for
   * device_do and implemented in sequential_enumerate.h.
   */
  template<typename F, typename... Args>
  __host_or_device__ void enumerate(F func, Args... args);

  /**
   * The number of bits per container.
   */
  static const uint8_t kBitsize = 8*sizeof(ContainerT);

  /**
   * The number of containers on this level.
   */
  static const SizeT kNumContainers =
      N <= kBitsize ? 1 : (N + kBitsize - 1)/kBitsize;

  /**
   * Indicates if this bitmap has a higher-level (nested) bitmap.
   */
  static const bool kHasNested = kNumContainers > 1;

  // Type aliases that can be accessed from outside this class.
  using SizeTT = SizeT;
  using ContainerTT = ContainerT;

  /**
   * Nested bitmap structure type.
   */
  using BitmapDataT = BitmapData<kNumContainers, kHasNested>;

  /**
   * Nested bitmap structure.
   */
  BitmapDataT data_;

  /**
   * Number of bitmap levels, including this one.
   */
  static const int kLevels = BitmapDataT::kLevels;

  /**
   * Type of outer bitmap. (Going the other way in the hierarchy.)
   */
  using OuterBitmapT = Bitmap<SizeT, N*kBitsize, ContainerT, ScanType>;
};

#include "bitmap/sequential_enumerate.h"

#endif  // BITMAP_BITMAP_H
