#ifndef ALLOCATOR_SOA_ALLOCATOR_H
#define ALLOCATOR_SOA_ALLOCATOR_H

#include <assert.h>
#include <tuple>
#include <typeinfo>
#include <type_traits>

#define __DEV__ __device__

#include "bitmap/bitmap.h"

#include "allocator/soa_base.h"
#include "allocator/soa_block.h"
#include "allocator/soa_defrag_record.h"
#include "allocator/soa_executor.h"
#include "allocator/soa_field.h"
#include "allocator/tuple_helper.h"
#include "allocator/util.h"
#include "allocator/type_system.h"


/**
 * This is a device-side memory allocator. Objects of this class reside in the
 * GPU memory. Objects of this class can be used to:
 * - Allocate an object of a type among \p Types with placement-new syntax.
 * - Deallocate an object that was previously allocated with the same
 *   device-side memory allocator.
 * - Initiate a device_do iteration.
 * - Retrieve various kinds of debug information/allocator statistics.
 * @tparam N_Objects The maximum number of objects of the smallest type (within
 *                   \p Types) that can be allocated.
 * @tparam Types A list (variadic template) of all types (structs/classes) that
 *               are under the control of this allocator.
 */
template<BlockIndexT N_Objects, class... Types>
class SoaAllocator {
 public:
  /**
   * An alias of this allocator type. Just for convenience.
   */
  using ThisAllocator = SoaAllocator<N_Objects, Types...>;

  /**
   * Classes/structs that are under control of this allocator type should
   * inherit from this class. They do not have to be a direct subclass of this
   * class, but the inheritance hierarchy should termimate with this class.
   */
  using Base = SoaBase<ThisAllocator>;

  /**
   * The maximum number of objects per block. Currently fixed at 64.
   */
  static const ObjectIndexT kNumBlockElements = 64;
  static_assert(kNumBlockElements == 64,
                "Not implemented: Block size != 64.");

  /**
   * A bitmask of bits that contain the block address within a fake pointer.
   */
  static const uint64_t kBlockAddrBitmask = PointerHelper::kBlockPtrBitmask;

  /**
   * The number of blocks that the heap consists of.
   */
  static const BlockIndexT N = N_Objects / kNumBlockElements;

  /**
   * An alias for the block state bitmap type. Just for convenience.
   */
  using StateBitmapT = Bitmap<BlockIndexT, N>;

  // N_Objects must be a multiple of 64.
  static_assert(N_Objects % kNumBlockElements == 0,
                "N_Objects Must be divisible by BlockSize.");


// Is memory defragmentation activated? (Check CompactGpu ISMM paper.)
#ifdef OPTION_DEFRAG
  // ---- Defragmentation (soa_defrag.inc) ----
  template<typename T, int NumRecords>
  __DEV__ void defrag_choose_source_block(int min_remaining_records);

  template<typename T, int NumRecords>
  __DEV__ void defrag_choose_target_blocks();

  template<typename DefragT, int NumRecords, typename ScanClassT,
           typename FieldT>
  __DEV__ void maybe_rewrite_pointer(FieldT* scan_location);

// Are we using forwarding pointers to rewrite pointers?
#ifdef OPTION_DEFRAG_FORWARDING_POINTER
  template<typename T>
  __DEV__ void defrag_update_block_state();

  template<typename T>
  __DEV__ void defrag_clear_source_leq_50();

  template<typename T>
  __DEV__ BlockIndexT get_num_defrag_compactions();

  template<typename T>
  __DEV__ BlockIndexT get_defrag_candidate_index(int did, int idx);

  template<typename T>
  __DEV__ void defrag_move();

  template<typename T>
  __DEV__ void defrag_store_forwarding_ptr();
#else
  template<typename T, int NumRecords>
  __DEV__ void defrag_move();

  template<typename T, int NumRecords>
  __DEV__ void defrag_update_block_state();
#endif  // OPTION_DEFRAG_FORWARDING_POINTER

  template<int NumRecords>
  __DEV__ void load_records_to_shared_mem();

  void DBG_print_defrag_time();

  // Should be invoked from host side.
  template<typename T, int NumRecords>
  void parallel_defrag(int min_num_compactions);

  template<typename T>
  void parallel_defrag(int min_num_compactions);
  // ---- END ----
#endif  // OPTION_DEFRAG

  // ---- Debugging (soa_debug.inc) ----
  /**
   * Counts the number of allocated allocated object slots of type \p T, i.e.,
   * slots of allocated blocks that do not contain an object.
   */
  template<class T>
  __DEV__ BlockIndexT DBG_allocated_slots();

  /**
   * Counts the number used object slots of type \p T.
   */
  template<class T>
  __DEV__ BlockIndexT DBG_used_slots();

  /**
   * A wrapper around DBG_allocated_slots() that can be called from host code.
   */
  template<class T>
  BlockIndexT DBG_host_allocated_slots();

  /**
   * A wrapper around DBG_used_slots() that can be called from host code.
   */
  template<class T>
  BlockIndexT DBG_host_used_slots();

  /**
   * Prints statistics about the types of the allocator, including detailed
   * information for about the structure of blocks.
   *
   * The following printing is produced by the wa-tor example program.
\verbatim
┌───────────────────────────────────────────────────────────────────────┐
│ Smallest block type:                                            4Fish │
│ Max. #objects:             262144                                     │
│ Block size:                  3904 bytes                               │
│ #Blocks:                     4096                                     │
│ #Bitmap levels:                 2                                     │
│ Data buffer size:     000015.250000 MB                                │
│ Allocator overead:    000000.114670 MB + block overhead               │
│ Total memory usage:   000015.364670 MB                                │
└───────────────────────────────────────────────────────────────────────┘
\endverbatim
   * This box gives an overview of the state of the allocator.
   * - The smallest type under control of this allocator is the C++ class Fish.
   * - The size of the heap was chosen such that 262144 objects of class Fish
   *   can be allocated.
   * - The size of a block (bitmaps, data segment, etc.) is 3904 bytes. This
   *   value corresponds to SoaAllocator::kBlockSizeBytes.
   * - The heap consists of 4096 blocks. This value corresponds to
   *   SoaAllocator::N.
   * - Each block state bitmap has 2 levels.
   * - The size of the data buffer (all blocks) is 15.25 MiB.
   * - The overhead of the allocator (all block state bitmaps) is 0.11 MiB.
   * - The total GPU memory consumption of the allocator is 15.36 MiB.
   *
\verbatim
┌───────────────────────────────────────────────────────────────────────┐
│ Block stats for                                5Shark (type ID     2) │
├────────────────────┬──────────────────────────────────────────────────┤
│ #fields            │        5                                         │
│ #objects / block   │       60                                         │
│ block size         │     3904 bytes                                   │
│ base class         │                                           5Agent │
│ is abstract        │        0                                         │
│ data seg. [60] sz  │     3840 bytes                                   │
│         (unpadded) │     3840 bytes                                   │
│        (simple sz) │     3840 bytes                                   │
│    (padding waste) │        0 bytes                                   │
│ data seg. [ 1] sz  │       64 bytes                                   │
│         (unpadded) │       64 bytes                                   │
│ data seg. [64] sz  │     4096 bytes                                   │
│         (unpadded) │     4096 bytes                                   │
├────────────────────┴──────────────────────────────────────────────────┤
│ Fields                                                                │
├───────┬─────────────────┬───────────────────────┬──────────┬──────────┤
│ Index │ Def. Class      │ Type                  │ Size     │ Offset   │
├───────┼─────────────────┼───────────────────────┼──────────┼──────────┤
│     1 │          5Shark │                     j │        4 │       60 │
│     0 │          5Shark │                     j │        4 │       56 │
│     2 │          5Agent │                     i │        4 │       52 │
│     1 │          5Agent │                     i │        4 │       48 │
│     0 │          5Agent │   17curandStateXORWOW │       48 │        0 │
├───────┼─────────────────┼───────────────────────┼──────────┼──────────┤
│     Σ │                 │                       │       64 │          │
└───────┴─────────────────┴───────────────────────┴──────────┴──────────┘
\endverbatim
   * There is a box with block structure information for each type that is
   * under control of the allocator.
   * - This box shows information about blocks of C++ class Shark.
   * - This class has 3 fields.
   * - Block of this type have a size of 3904 bytes. Note that the block size
   *   of each type is always smaller or equal to 3904, as shown above.
   * - This class inherits from class Agent.
   * - This class is not abstract.
   * - The data segment of blocks of this type have a size of 3840 bytes.
   * - If "(unpadded)" is smaller than "data seg. [x] sz", then at least one
   *   SOA array was padded such that the array is properly aligned.
   * - The box also shows detailed information about each field, including
   *   fields that were inherited from class Agent. Primitive types are
   *   currently not properly printed. E.g., "j" means float and "i" means int.
   */
  static void DBG_print_stats();

  /**
   * Calculates the overall fragmentation rate of the heap. The fragmentation
   * rate is defined as the fraction of allocated but unused object slots
   * among all allocated object slots. See ECOOP paper for details.
   */
  __DEV__ float DBG_calculate_fragmentation();

  /**
   * Prints detailed statistics about the state of the allocator and the heap.
\verbatim
┌────┬──────────┬──────────┬──────────┬┬──────────┬──────────┬──────────┐
│ Ty │ #B_alloc │ #B_leq50 │ #B_activ ││ #O_alloc │  #O_used │   O_frag │
├────┼──────────┼──────────┼──────────┼┼──────────┼──────────┼──────────┤
│ fr │     1320 │      n/a │      n/a ││      n/a │      n/a │      n/a │
│  0 │        0 │        0 │        0 ││        0 │        0 │ 0.000000 │
│  1 │     2224 │        0 │     2141 ││   142336 │   133631 │ 0.061158 │
│  2 │      552 │        0 │      411 ││    33120 │    12478 │ 0.623249 │
│  Σ │     2776 │        0 │     2552 ││   175456 │   146109 │ 0.167261 │
└────┴──────────┴──────────┴──────────┴┴──────────┴──────────┴──────────┘
\endverbatim
   * In the above example, there is one line for each type that is under
   * control of the allocator.
   * - B_alloc: Number of allocated blocks
   * - B_leq50: Number of defragmentation candidates
   * - B_activ: Number of active blocks
   * - O_alloc: Number of allocated object slots
   * - O_used:  Number of used object slots. Always >= O_alloc.
   * - O_frag:  Fragmentation rate: 1 - (O_used / O_alloc)
   * - Type fr: Free blocks
   */
  __DEV__ void DBG_print_state_stats();

  /**
   * Can be called after each parallel do-all operation to determine the
   * maximum heap usage and fragmentation rate throughout the application
   * runtime.
   */
  __DEV__ void DBG_collect_stats();

  /**
   * Print statistics that were gathered by DBG_collect_stats().
   */
  __DEV__ void DBG_print_collected_stats();

  /**
   * A wrapper around DBG_calculate_fragmentation() that can be called from
   * host code.
   */
  float DBG_host_calculate_fragmentation();

  /**
   * Returns the time spent on enumeration during parallel do-all operations
   * (e.g., bitmap scan). Parallel do-all automatically keeps track of this
   * time.
   */
  long unsigned int DBG_get_enumeration_time() const {
    return bench_prefix_sum_time;
  }
  // ---- END ----


  /**
   * A helper class that provides useful aliases and constants for blocks of
   * a certain type.
   * @tparam T Block type
   */
  template<typename T>
  struct BlockHelper {
    // SoaBase<> has size 1, everything else size 0.
    static_assert(sizeof(T) == 1,
                  "Unexpected superclass size.");

    /**
     * Index / type ID of \param T within this allocator.
     */
    static const int kIndex = TYPE_INDEX(Types..., T);

    /**
     * Block capacity (max. number of objects per block).
     */
    static const int kSize =
        SoaBlockSizeCalculator<T, kNumBlockElements, TupleHelper<Types...>
            ::k64BlockMinSize>::kSize;

    /**
     * Data segment size in bytes.
     */
    static const int kBytes =
        SoaBlockSizeCalculator<T, kNumBlockElements, TupleHelper<Types...>
            ::k64BlockMinSize>::kBytes;

    /**
     * Block type alias.
     */
    using BlockType = SoaBlock<T, kIndex, kSize>;

#ifdef OPTION_DEFRAG
    /**
     * Defragmentation candidate threshold. If a block that more object then
     * it is no longer a defragmentation candidate.
     */
    static const int kLeq50Threshold = BlockType::kLeq50Threshold;
#endif  // OPTION_DEFRAG
  };

  /**
   * 64-bit data type for bitmaps within a block (e.g., object allocation
   * bitmap).
   */
  using BlockBitmapT = unsigned long long int;

  /**
   * A helper class that determines the type ID of a type \p T.
   */
  template<typename T>
  struct TypeId {
    static const TypeIndexT value = BlockHelper<T>::kIndex;
  };

  /**
   * Extracts the type ID from a fake pointer and returns it.
   * @param ptr Fake pointer
   */
  __device__ __host__ static TypeIndexT get_type(const void* ptr) {
    auto type_id = PointerHelper::get_type(ptr);
    assert(type_id < kNumTypes);
    return type_id;
  }

  /**
   * Checks if the type encoded in the fake pointer \p ptr is \p T.
   * @param ptr Fake pointer
   * @tparam T Expected type
   */
  template<class T>
  __device__ __host__ static bool is_type(const T* ptr) {
    return TupleHelper<Types...>::template dev_for_all<PointerTypeChecker<
        ThisAllocator, T>::template InnerHelper>(ptr);
  }

  __DEV__ SoaAllocator(const ThisAllocator&) = delete;

  /**
   * Initializes the allocator (device memory).
   * - All bits in the free block bitmap are initially 1.
   * - All bits in the allocated/active block bitmaps are initially 0.
   * - All bits in the defrag. candidate bitmaps are initially 0.
   * @param data_buffer The data buffer that contains the heap (excl. block
                        state bitmaps).
   */
  __DEV__ SoaAllocator(char* data_buffer) : data_(data_buffer), global_free_(true) {
    for (int i = 0; i < kNumTypes; ++i) {
      new(allocated_ + i) StateBitmapT(false);
      new(active_ + i) StateBitmapT(false);

#ifdef OPTION_DEFRAG
      new(leq_50_ + i) StateBitmapT(false);
      num_leq_50_[i] = 0;
#endif  // OPTION_DEFRAG
    }

    // Check alignment of data storage buffer.
    assert(reinterpret_cast<uintptr_t>(data_) % 64 == 0);

    // Ensure that most significant bits of data address are not in use. This
    // is required for our fake pointer mangling scheme.
    assert((reinterpret_cast<uintptr_t>(data_)
            & ~PointerHelper::kMemAddrBitmask) == 0);
    assert(((reinterpret_cast<uintptr_t>(data_) + kDataBufferSize)
            & ~PointerHelper::kMemAddrBitmask) == 0);
  }

  /**
   * Allocates a new object of type \p T and returns a fake pointer to the
   * object.
   */
  template<class T>
  __DEV__ T* allocate_new() {
    T* result = nullptr;

    do {
      const auto active = __activemask();
      // Leader thread is the first thread whose mask bit is set to 1.
      const auto leader = __ffs(active) - 1;
      assert(leader >= 0 && leader < 32);
      // Use lane mask to empty all bits higher than the current thread.
      // The rank of this thread is the number of bits set to 1 in the result.
      const auto rank = __lane_id();
      assert(rank < 32);

      // Values to be calculated by the leader.
      BlockIndexT block_idx;
      BlockBitmapT allocation_bitmap;
      if (rank == leader) {
        assert(__popc(__activemask()) == 1);    // Only one thread executing.

        block_idx = find_active_block<T>();
        auto* block = get_block<T>(block_idx);
        BlockBitmapT* free_bitmap = &block->free_bitmap;
        allocation_bitmap = allocate_in_block<T>(
            free_bitmap, __popc(active), block_idx);
        auto num_allocated = __popcll(allocation_bitmap);

        auto actual_type_id = block->type_id;
        if (actual_type_id != BlockHelper<T>::kIndex) {
          // TODO: Check correctness. This code is rarely executed.

          // Block deallocated and initialized for a new type between lookup
          // from active bitmap and here. This is extremely unlikely!
          // But possible.
          // Undo allocation and update bitmaps accordingly.

          // Note: Cannot be in invalidation here, because we are certain that
          // we allocated the bits that we are about to deallocate here.
          auto before_undo = atomicOr(free_bitmap, allocation_bitmap);
          auto slots_before_undo = __popcll(before_undo);

#ifdef OPTION_DEFRAG
          if (BlockHelper<T>::kSize - slots_before_undo
                  > BlockHelper<T>::kLeq50Threshold
              && BlockHelper<T>::kSize - slots_before_undo - num_allocated
                  <= BlockHelper<T>::kLeq50Threshold) {
            ASSERT_SUCCESS(leq_50_[actual_type_id].allocate<true>(block_idx));
            atomicAdd(&num_leq_50_[actual_type_id], 1);
          }
#endif  // OPTION_DEFRAG

          // Cases to handle: block now active again or block empty now.
          if (slots_before_undo == 0) {
            // Block became active. (Was full.)
            ASSERT_SUCCESS(active_[actual_type_id].allocate<true>(block_idx));
          } else if (slots_before_undo == BlockHelper<T>::kSize - 1) {
            // Block now empty.
            if (invalidate_block<T>(block_idx)) {
              // Block is invalidated and no new allocations can be performed.
              ASSERT_SUCCESS(active_[actual_type_id].deallocate<true>(block_idx));
#ifdef OPTION_DEFRAG
              ASSERT_SUCCESS(leq_50_[actual_type_id].deallocate<true>(block_idx));
              atomicSub(&num_leq_50_[actual_type_id], 1);
#endif // OPTION_DEFRAG
              ASSERT_SUCCESS(allocated_[actual_type_id].deallocate<true>(block_idx));
              ASSERT_SUCCESS(global_free_.allocate<true>(block_idx));
            }
          }
        }
      }

      assert(__activemask() == active);
      // Get pointer from allocation (nullptr if no allocation).
      allocation_bitmap = __shfl_sync(active, allocation_bitmap, leader);
      block_idx = __shfl_sync(active, block_idx, leader);
      assert(block_idx < N);
      result = get_ptr_from_allocation<T>(
          block_idx, __popc(__lanemask_lt() & active), allocation_bitmap);
    } while (result == nullptr);

    assert(PointerHelper::obj_id_from_obj_ptr(result) < BlockHelper<T>::kSize);
    return result;
  }

  /**
   * Allocates a new object of type \p T, runs the constructor and returns a
   * fake pointer to the object. This is an alternative for object construction
   * with the "new" keyword.
   */
  template<typename T, typename... Args>
  __DEV__ T* make_new(Args... args) {
    return new(allocate_new<T>()) T(std::forward<Args>(args)...);
  }

  /**
   * Deallocates an object of type \p T or a subtype of \p T, given a fake
   * pointer to the object. This function first determines the runtime type of
   * the object and then calls free_typed<R>() with the runtime type R of
   * \p obj.
   */
  template<class T>
  __DEV__ void free(T* obj) {
    auto type_id = obj->get_type();
    if (type_id == BlockHelper<T>::kIndex) {
      free_typed(obj);
    } else {
      ASSERT_SUCCESS(TupleHelper<Types...>
          ::template dev_for_all<FreeHelper<T>::InnerHelper>(this, obj));
    }
  }

  /**
   * Deallocates an object given a fake pointer and the exact runtime type ID
   * as a template parameter \p TypeIndex.
   */
  template<int TypeIndex>
  __DEV__ void free_untyped(void* obj) {
    auto* typed = static_cast<TYPE_ELEMENT(Types, TypeIndex)*>(obj);
    free(typed);
  }

  /**
   * Parallel do-all: Run a member function \p func of class/struct \p T on all
   * objects of type \p IterT, not taking into account subclasses of \p IterT.
   * The parameter \p Scan indicates whether this operation should be preceded
   * by a bitmap scan operation.
   * - Spawns a CUDA kernel.
   * - Enumerates only objects that exist at invocation time.
   * @tparam IterT Enumerate objects of this type.
   * @tparam T Class in which the function is defined.
   * @tparam func Function to be run for each object.
   * @tparam Scan Perform bitmap scan?
   */
  template<class IterT, class T, void(T::*func)(), bool Scan>
  void parallel_do_single_type() {
    ParallelExecutor<Scan, ThisAllocator, IterT, T>
        ::template FunctionArgTypesWrapper<void, T>
        ::template FunctionWrapper<func>
        ::parallel_do(this, /*shared_mem_size=*/ 0);
  }

  /**
   * Parallel do-all: Run a member function \p func of class/struct \p T on all
   * objects of type \p IterT, not taking into account subclasses of \p IterT.
   * The parameter \p Scan indicates whether this operation should be preceded
   * by a bitmap scan operation. \p func takes an argument of type \p P1.
   * See parallel_do_single_type() for details.
   * TODO: Generate versions with more than one argument with template pattern
   *       matching.
   * @tparam IterT Enumerate objects of this type.
   * @tparam T Class in which the function is defined.
   * @tparam func Function to be run for each object.
   * @tparam P1 Type of parameter of \p func.
   * @tparam Scan Perform bitmap scan?
   */
  template<class IterT, class T, typename P1, void(T::*func)(P1), bool Scan>
  void parallel_do_single_type(P1 p1) {
    ParallelExecutor<Scan, ThisAllocator, IterT, T>
        ::template FunctionArgTypesWrapper<void, T, P1>
        ::template FunctionWrapper<func>
        ::parallel_do(this, /*shared_mem_size=*/ 0, std::forward<P1>(p1));
  }

  /**
   * Parallel do-all: Run a member function \p func of class/struct \p T on all
   * objects of type \p T, also taking into account subclasses of \p T.
   * The parameter \p Scan indicates whether this operation should be preceded
   * by a bitmap scan operation.
   * @tparam T Class in which the function is defined.
   * @tparam func Function to be run for each object.
   * @tparam Scan Perform bitmap scan?
   */
  template<bool Scan, class T, void(T::*func)()>
  void parallel_do() {
    TupleHelper<Types...>
        ::template for_all<ParallelDoTypeHelperL1<>
        ::template ParallelDoTypeHelperL2<ThisAllocator, T, func, Scan>
        ::template ParallelDoTypeHelperL3>(this);
  }

  /**
   * Parallel do-all: Run a member function \p func of class/struct \p T on all
   * objects of type \p T, also taking into account subclasses of \p T.
   * The parameter \p Scan indicates whether this operation should be preceded
   * by a bitmap scan operation. \p func takes an argument of type \p P1.
   * TODO: Generate versions with more than one argument with template pattern
   *       matching.
   * @tparam T Class in which the function is defined.
   * @tparam func Function to be run for each object.
   * @tparam P1 Type of parameter of \p func.
   * @tparam Scan Perform bitmap scan?
   */
  template<bool Scan, class T, typename P1, void(T::*func)(P1)>
  void parallel_do(P1 p1) {
    TupleHelper<Types...>
        ::template for_all<ParallelDoTypeHelperL1<P1>
        ::template ParallelDoTypeHelperL2<ThisAllocator, T, func, Scan>
        ::template ParallelDoTypeHelperL3>(this, std::forward<P1>(p1));
  }

  /**
   * Device do: Run a member function \p func on all objects of type \p in the
   * current GPU thread.
   * TODO: This function should also enumerate subtypes and omit newly created
   * objects.
   * @tparam T Class of objects that should be enumerated.
   * @tparam F Function to be run for each object.
   * @tparam Args Types of parameters of \p F.
   */
  template<class T, typename F, typename... Args>
  __host_or_device__ void device_do(F func, Args... args) {
    // device_do iterates over objects in a block.
    allocated_[BlockHelper<T>::kIndex].enumerate(
        &SequentialExecutor<T, F, ThisAllocator, Args...>::device_do,
        func, this, std::forward<Args>(args)...);
  }

  /**
   * Initializes a parallel do-all operation. This function should be run
   * before each parallel do-all. It sets the object iteration bitmap of every
   * allocated block of type \p T to its object allocation bitmap.
   * @tparam T Class of objects that should be enumerated.
   */
  template<typename T>
  __DEV__ void initialize_iteration() {
    const auto num_blocks = allocated_[BlockHelper<T>::kIndex].scan_num_bits();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_blocks;
         i += blockDim.x * gridDim.x) {
      auto idx = allocated_[BlockHelper<T>::kIndex].scan_get_index(i);
      assert(allocated_[BlockHelper<T>::kIndex][idx]);

      // Initialize block for iteration.
      get_block<T>(idx)->initialize_iteration();
    }
  }

  /**
   * Reserves one or multiple object slots in a block. Due to allocation
   * request coalescing, this function is executed by only one GPU thread.
   * @tparam T Type of objects to be allocated.
   * @param free_bitmap_ptr A pointer to the object allocation bitmap of the
   *                        block. Note: DynaSOAr actually maintains free
   *                        bitmaps instead of allocation bitmaps.
   * @param alloc_size The number of objects to be allocated (in the warp).
   * @param block_idx The index of the block in which objects are allocated.
   */
  template<typename T>
  __DEV__ BlockBitmapT allocate_in_block(BlockBitmapT* free_bitmap_ptr,
                                         int alloc_size, BlockIndexT block_idx) {
    // Allocation bits.
    BlockBitmapT selected_bits = 0;
    // Set to true if block is full.
    bool block_full;

    BlockBitmapT free_bitmap = *free_bitmap_ptr;

    do {
      // Bit set to 1 if slot is free.
      // TODO: Try different ones.
      const auto rotation_len = warp_id() % 64;
      // TODO: Can we use return value from atomic update in second iteration?
      BlockBitmapT updated_mask = rotl(free_bitmap, rotation_len);

      // If there are not enough free slots, allocate as many as possible.
      auto free_slots = __popcll(updated_mask);
      auto allocation_size = min(free_slots, alloc_size);
      BlockBitmapT newly_selected_bits = 0;

      // Generate bitmask for allocation
      for (int i = 0; i < allocation_size; ++i) {
        auto next_bit_pos = __ffsll(updated_mask) - 1;
        assert(next_bit_pos >= 0);
        assert(((1ULL << next_bit_pos) & updated_mask) > 0);
        // Clear bit at position `next_bit_pos` in updated mask.
        updated_mask &= updated_mask - 1;
        // Save location of selected bit.
        auto next_bit_pos_unrot = (next_bit_pos - rotation_len) % 64;
        newly_selected_bits |= 1ULL << next_bit_pos_unrot;
      }

      assert(__popcll(newly_selected_bits) == allocation_size);
      // Count the number of bits that were selected but already set to false
      // by another thread.
      BlockBitmapT before_update = atomicAnd(free_bitmap_ptr, ~newly_selected_bits);
      free_bitmap = before_update & ~newly_selected_bits;
      BlockBitmapT successful_alloc = newly_selected_bits & before_update;
      block_full = (before_update & ~successful_alloc) == 0;

      if (successful_alloc > 0ULL) {
        // At least one slot allocated.
        auto num_successful_alloc = __popcll(successful_alloc);
        alloc_size -= num_successful_alloc;
        selected_bits |= successful_alloc;

        if (block_full) {
          ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
        }

#ifdef OPTION_DEFRAG
        // Check if more than 50% full now.
        auto prev_full = BlockHelper<T>::kSize - __popcll(before_update);
        if (prev_full <= BlockHelper<T>::kLeq50Threshold
            && prev_full + num_successful_alloc > BlockHelper<T>::kLeq50Threshold) {
          ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
          atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
        }
#endif  // OPTION_DEFRAG
      }

      // Stop loop if no more free bits available in this block or all
      // requested allocations completed successfully.
    } while (alloc_size > 0 && !block_full);

    // At most one thread should indicate that the block filled up.
    return selected_bits;
  }

  /**
   * Deallocates a block, i.e., turns the block into a free block. This
   * function merely updates block state bitmaps and assumes that the block
   * is already invalidated. By default, it assumes that the block is a
   * defragmentation candidate.
   * @tparam T Type of the block.
   * @param block_idx The index of the block.
   * @param dealloc_leq_50 Update defragmentation candidates bitmap?
   */
  template<class T>
  __DEV__ void deallocate_block(BlockIndexT block_idx, bool dealloc_leq_50 = true) {
    // Precondition: Block is invalidated.
    assert(get_block<T>(block_idx)->free_bitmap == 0);
    ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));

#ifdef OPTION_DEFRAG
    if (dealloc_leq_50) {
      ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
      atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
    }
#endif  // OPTION_DEFRAG

    ASSERT_SUCCESS(allocated_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
    ASSERT_SUCCESS(global_free_.allocate<true>(block_idx));
  }

  /**
   * Deallocates an object with a given fake pointer. This function assumes
   * that the exact runtime type of the object is \p T (not a subtype).
   * @tparam T Runtime type of the object.
   * @param obj Fake pointer to the object.
   */
  template<class T>
  __DEV__ void free_typed(T* obj) {
    obj->~T();
    const auto block_idx = get_block_idx<T>(obj);
    const auto obj_id = get_object_id<T>(obj);
    const auto dealloc_state = get_block<T>(block_idx)->deallocate(obj_id);

    // Note: Different ordering of branches can lead to deadlock!
    if (dealloc_state == kBlockNowActive) {
      ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
#ifdef OPTION_DEFRAG
    } else if (dealloc_state == kBlockNowLeq50Full) {
      ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
      atomicAdd(&num_leq_50_[BlockHelper<T>::kIndex], 1);
#endif  // OPTION_DEFRAG
    } else if (dealloc_state == kBlockNowEmpty) {
      // Assume that block is empty.
      // TODO: Special case if N == 2 or N == 1 for leq_50_.
      if (invalidate_block<T>(block_idx)) {
        // Block is invalidated and no new allocations can be performed.
        deallocate_block<T>(block_idx, /*dealloc_leq_50=*/ true);
      }
    }
  }

  // Helper data structure for freeing objects whose types are subtypes of the
  // declared type. BaseClass is the declared type.
  template<typename BaseClass>
  struct FreeHelper {
    // Iterating over all types T in the allocator.
    template<typename T>
    struct InnerHelper {
      // T is a subclass of BaseClass. Check if same type.
      template<bool Check, int Dummy>
      struct ClassSelector {
        __DEV__ static bool call(ThisAllocator* allocator, BaseClass* obj) {
          if (obj->get_type() == BlockHelper<T>::kIndex) {
            allocator->free_typed(static_cast<T*>(obj));
            return false;  // No need to check other types.
          } else {
            return true;   // true means "continue processing".
          }
        }
      };

      // T is not a subclass of BaseClass. Skip.
      template<int Dummy>
      struct ClassSelector<false, Dummy> {
        __DEV__ static bool call(ThisAllocator* allocator, BaseClass* obj) {
          return true;
        }
      };

      __DEV__ bool operator()(ThisAllocator* allocator, BaseClass* obj) {
        return ClassSelector<std::is_base_of<BaseClass, T>::value, 0>::call(
            allocator, obj);
      }
    };
  };

  /**
   * Checks if a block location is in state allocated[T].
   */
  template<typename T>
  __device__ __host__ bool is_block_allocated(BlockIndexT index) {
    return allocated_[BlockHelper<T>::kIndex][index];
  }

  /**
   * Decodes the block index from a fake pointer.
   */
  template<class T>
  __device__ __host__ BlockIndexT get_block_idx(T* ptr) {
    uintptr_t ptr_as_int = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t data_as_int = reinterpret_cast<uintptr_t>(data_);

    assert(((ptr_as_int & kBlockAddrBitmask) - data_as_int) % kBlockSizeBytes == 0);
    return ((ptr_as_int & kBlockAddrBitmask) - data_as_int) / kBlockSizeBytes;
  }

  /**
   * Decodes the object slot ID from a fake pointer.
   */
  template<class T>
  __device__ __host__ static ObjectIndexT get_object_id(T* ptr) {
    return PointerHelper::obj_id_from_obj_ptr(ptr);
  }

  /**
   * Build a fake pointer for an object of type \p T in a given block and with
   * a given object slot ID.
   * @tparam T Object type
   * @param block Pointer to block
   * @param obj_id Object slot ID
   */
  template<class T>
  __device__ __host__ static T* get_object(
      typename BlockHelper<T>::BlockType* block, ObjectIndexT obj_id) {
    assert(obj_id < 64);
    return block->make_pointer(obj_id);
  }

  /**
   * Get a pointer to a block with a given index.
   * @tparam Object type
   * @param block_idx Block index
   */
  template<class T>
  __device__ __host__ typename BlockHelper<T>::BlockType* get_block(
      BlockIndexT block_idx) const {
    assert(block_idx < N && block_idx >= 0);
    uintptr_t increment = static_cast<uintptr_t>(block_idx)*kBlockSizeBytes;
    auto* result = reinterpret_cast<typename BlockHelper<T>::BlockType*>(
        data_ + increment);
    assert(reinterpret_cast<char*>(result) >= data_);
    return result;
  }

  /**
   * Finds and returns the index of a block in state active[T]. If no block
   * could be found, this function retries, until up to
   * \p kFindActiveBlockRetries attempts were made. If still no block could be
   * found, this function initializes a new active[T] block.
   * - This function must be used with caution because, by the time the
   *   function returns, the returned block may already not longer be active.
   * - When running out of memory, this function does not terminate and runs
   *   indefinitely. TODO: Change this.
   */
  template<class T>
  __DEV__ BlockIndexT find_active_block() {
    BlockIndexT block_idx;

    do {
      // Retry a couple of times. May reduce fragmentation.
      int retries = kFindActiveBlockRetries;   // retries=2 before
      do {
        block_idx = active_[BlockHelper<T>::kIndex]
            .template find_allocated<false>(retries + blockIdx.x);
      } while (block_idx == StateBitmapT::kIndexError && --retries > 0);

      if (block_idx == StateBitmapT::kIndexError) {
        // TODO: May be out of memory here.
        block_idx = global_free_.deallocate_seed(blockIdx.x);
        assert(block_idx != (StateBitmapT::kIndexError));  // OOM
        initialize_block<T>(block_idx);
        ASSERT_SUCCESS(allocated_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
#ifdef OPTION_DEFRAG
        ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
        atomicAdd(&num_leq_50_[BlockHelper<T>::kIndex], 1);
#endif  // OPTION_DEFRAG
        ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].allocate<true>(block_idx));
      }
    } while (block_idx == StateBitmapT::kIndexError);

    assert(block_idx < N);
    return block_idx;
  }

  /**
   * Initializes a new block of type \p T in a given block location.
   * @tparam T Object type
   * @param block_idx Index of block
   */
  template<class T>
  __DEV__ void initialize_block(BlockIndexT block_idx) {
    static_assert(sizeof(typename BlockHelper<T>::BlockType)
          % kNumBlockElements == 0,
        "Internal error: SOA block not aligned to 64 bytes.");
    assert(block_idx >= 0);
    auto* block_ptr = get_block<T>(block_idx);
    assert(reinterpret_cast<char*>(block_ptr) >= data_);
    new(block_ptr) typename BlockHelper<T>::BlockType();
  }

  /**
   * Builds a fake pointer from a bitmap of newly allocated object slots. This
   * function extracts the \p rank -th set bit index from \p allocation and
   * builds a fake pointer for that object slot in block \p block_idx and with
   * type \p T. Returns nullptr if there are not enough set bits.
   * @tparam T Object type
   * @param block_idx Block index
   * @param rank Set bit index
   * @param allocation A bitmap with set bits for newly allocated object slots
   */
  template<class T>
  __DEV__ T* get_ptr_from_allocation(BlockIndexT block_idx, int rank,
                                     BlockBitmapT allocation) {
    assert(block_idx < N);
    assert(rank < 32);

    // Get index of rank-th first bit set to 1.
    for (int i = 0; i < rank; ++i) {
      // Clear last bit.
      allocation &= allocation - 1;
    }

    auto position = __ffsll(allocation);

    if (position > 0) {
      // Allocation successful.
      return get_block<T>(block_idx)->make_pointer(position - 1);
    } else {
      return nullptr;
    }
  }

  /**
   * Invalidates a block.
   * - Precondition: The block must be active.
   * - Postcondition: If the block was successfully invalidated, then its type
   *   can no longer change until it is reinitialized.
   *
   * @tparam T Type of block. TODO: Block invalidation should be independent
   *           of the block type.
   * @param block_idx Index of the block
   */
  template<class T>
  __DEV__ bool invalidate_block(BlockIndexT block_idx) {
    auto* block = get_block<T>(block_idx);

    while (true) {
      auto old_free_bitmap = atomicExch(&block->free_bitmap, 0ULL);
      if (old_free_bitmap == BlockHelper<T>::BlockType::kBitmapInitState) {
        return true;
      } else if (old_free_bitmap != 0ULL) {
        // TODO: Check correctness of this code path. It is rarely executed.

        // block->free_bitmap = old_free_bitmap;

        // free_bitmap now 0. We should deactivate the block here, but since
        // the block deallocation procedure expects the invalid block to be
        // active, we omit this and only deactivate it in the specical case
        // described below.

        // At least one bit was modified. Rollback invalidation.
        auto before_rollback = atomicOr(&block->free_bitmap, old_free_bitmap);
        if (before_rollback > 0ULL) {
#ifdef OPTION_DEFRAG
          if ((BlockHelper<T>::kSize - __popcll(old_free_bitmap)
                  > BlockHelper<T>::kLeq50Threshold)
              && (BlockHelper<T>::kSize - __popcll(before_rollback)
                  <= BlockHelper<T>::kLeq50Threshold)) {
            // Some thread is trying to set the bit in the leq_50 bitmap.
            ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
            atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
          }
#endif  // OPTION_DEFRAG

          // At least 1 other thread deallocated an object (set a bit). That
          // thread is attempting to make the block active. For this to
          // succeed, we have to deactivate the block here.
          ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(block_idx));
        }

        if ((before_rollback | old_free_bitmap) !=
            BlockHelper<T>::BlockType::kBitmapInitState) {
          break;
        }  // else: Block emptied again. Try invalidating it again.
      }
    }

    return false;
  }

  /**
   * The number of types under control of this allocator.
   */
  static const int kNumTypes = sizeof...(Types);

  /**
   * The size of block in bytes.
   */
  static const int kBlockSizeBytes =
      sizeof(typename BlockHelper<typename TupleHelper<Types...>
             ::Type64BlockSizeMin>::BlockType);

  /**
   * Free block bitmap.
   */
  StateBitmapT global_free_;

  /**
   * Allocated block bitmaps (one per type).
   */
  StateBitmapT allocated_[kNumTypes];

  /**
   * Active block bitmaps (one per type).
   */
  StateBitmapT active_[kNumTypes];

#ifdef OPTION_DEFRAG
  /**
   * Defragmentation candidate bitmaps (one per type). A bit is set in this
   * bitmap if a block is <= 50% full (for defragmentation factor 1). We use a
   * CUB prefix sum for bitmap scans because it preserves the order of set bit
   * indices.
   */
  Bitmap<BlockIndexT, N, unsigned long long int, kCubScan> leq_50_[kNumTypes];

  /**
   * Number of defragmentation candidates (one counter per type). This counter
   * allows us to decide quickly if a defragmentation pass should be started.
   */
  BlockIndexT num_leq_50_[kNumTypes];

  /**
   * Temporary storage for defragmentation records.
   */
  SoaDefragRecords<BlockBitmapT, kMaxDefragRecords> defrag_records_;
#endif  // OPTION_DEFRAG

  /**
   * The heap: An array of blocks.
   */
  char* data_;

  /**
   * Size of the heap in bytes.
   */
  static const size_t kDataBufferSize = static_cast<size_t>(N)*kBlockSizeBytes;

  static const size_t kN = N;
};


template<typename T, typename F, typename... Args>
struct DeviceDoFunctionInvoker {
  // Pragma disables compiler warning.
  #pragma hd_warning_disable
  template<typename U = F>
  __host__ __device__ static typename
  std::enable_if<std::is_member_function_pointer<U>::value, void>::type
  call(T* obj, F func, Args... args) {
    (obj->*func)(std::forward<Args>(args)...);
  }

  #pragma hd_warning_disable
  template<typename U = F>
  __host__ __device__ static typename
  std::enable_if<!std::is_member_function_pointer<U>::value, void>::type
  call(T* obj, F func, Args... args) {
    func(obj, std::forward<Args>(args)...);
  }

  template<typename... Args2>
  __host__ __device__ static void call(Args2... args) {}
};


#pragma hd_warning_disable
template<typename T, typename F, typename AllocatorT, typename... Args>
__host__ __device__ void SequentialExecutor<T, F, AllocatorT, Args...>::device_do(
    BlockIndexT block_idx, F func, AllocatorT* allocator, Args... args) {
  auto* block = allocator->template get_block<T>(block_idx);
  auto bitmap = block->allocation_bitmap();

  while (bitmap != 0ULL) {
    auto pos = bit_ffsll(bitmap) - 1;
    bitmap &= bitmap - 1;

    auto* obj = AllocatorT::template get_object<T>(block, pos);
    DeviceDoFunctionInvoker<T, F, Args...>::call(
        obj, func, std::forward<Args>(args)...);
  }
}


// This are textual headers. Must be included at the end of the file.
#ifdef OPTION_DEFRAG
#include "allocator/soa_defrag.inc"
#endif  // OPTION_DEFRAG

#include "allocator/soa_debug.inc"

#endif  // ALLOCATOR_SOA_ALLOCATOR_H
