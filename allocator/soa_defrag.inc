// Textual header.

static const BlockIndexT kInvalidBlockIndex =
    std::numeric_limits<BlockIndexT>::max();

template<int NumBuckets, typename IndexT>
__DEV__ IndexT block_idx_hash(IndexT block_idx) {
  return block_idx % NumBuckets;
}


template<typename ClassIteratorT, typename AllocatorT>
__global__ void kernel_iterate_rewrite_objects(AllocatorT* allocator) {
  ClassIteratorT::iterate_rewrite_objects(allocator);
}


template<typename AllocatorT>
struct AllocatorWrapperDefrag {
  template<typename T>
  using BlockHelper = typename AllocatorT::template BlockHelper<T>;

  using BlockBitmapT = typename AllocatorT::BlockBitmapT;

  static const int kCudaBlockSize = 256;

  // Select fields of type DefragT* and rewrite pointers if necessary.
  // DefragT: Type that was defragmented.
  // ScanClassT: Class which is being scanned for affected fields.
  // SoaFieldHelperT: SoaFieldHelper of potentially affected field.
  template<typename DefragT, int NumRecords>
  struct SoaPointerUpdater {
    template<typename ScanClassT>
    struct ClassIterator {
      using ThisClass = ClassIterator<ScanClassT>;

      static const int kScanBlockSize = BlockHelper<ScanClassT>::kSize;
      static const int kScanTypeIndex = BlockHelper<ScanClassT>::kIndex;

      // Checks if this field should be rewritten.
      template<typename SoaFieldHelperT>
      struct FieldChecker {
        using FieldType = typename SoaFieldHelperT::type;

        bool operator()() {
          // Stop iterating if at least one field must be rewritten.
          return !(std::is_pointer<FieldType>::value
              && std::is_base_of<typename std::remove_pointer<FieldType>::type,
                                 DefragT>::value);
        }
      };

      __DEV__ static void iterate_rewrite_objects(AllocatorT* allocator) {
#ifndef OPTION_DEFRAG_FORWARDING_POINTER
        allocator->template load_records_to_shared_mem<NumRecords>();
#endif  // OPTION_DEFRAG_FORWARDING_POINTER

        const auto N_alloc =
            allocator->allocated_[kScanTypeIndex].scan_num_bits();

        // Round to multiple of kScanBlockSize.
        int num_threads = ((blockDim.x * gridDim.x)/kScanBlockSize)*kScanBlockSize;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < num_threads) {
          for (int j = tid/kScanBlockSize; j < N_alloc; j += num_threads/kScanBlockSize) {
            // i is the index of in the scan array.
            auto block_idx = allocator->allocated_[kScanTypeIndex].scan_get_index(j);

            // TODO: Consider doing a scan over "allocated" bitmap.
            auto* block = allocator->template get_block<ScanClassT>(block_idx);
            const auto allocation_bitmap = block->allocation_bitmap();
            int thread_offset = tid % kScanBlockSize;

            if ((allocation_bitmap & (1ULL << thread_offset)) != 0ULL) {
              SoaClassHelper<ScanClassT>
                  ::template dev_for_all<FieldUpdater, /*IterateBase=*/ true>(
                      allocator, block, thread_offset);
            }
          }
        }
      }

      // Scan and rewrite field.
      template<typename SoaFieldHelperT>
      struct FieldUpdater {
        using FieldType = typename SoaFieldHelperT::type;

        using SoaFieldType = SoaField<typename SoaFieldHelperT::OwnerClass,
                                      SoaFieldHelperT::kIndex>;

        // Scan this field.
        template<bool Check, int Dummy> 
        struct FieldSelector {
          template<typename BlockT>
          __DEV__ static void call(AllocatorT* allocator,
                                   BlockT* block, ObjectIndexT object_id) {
#ifndef OPTION_DEFRAG_FORWARDING_POINTER
            extern __shared__ DefragRecord<BlockBitmapT> records[];
#endif  // OPTION_DEFRAG_FORWARDING_POINTER

            // Location of field value to be scanned/rewritten.
            auto* block_char_p = reinterpret_cast<char*>(block);
            FieldType* scan_location =
                SoaFieldType::template data_ptr_from_location<BlockT::kN>(
                    block_char_p, object_id);

            assert(reinterpret_cast<char*>(scan_location) >= allocator->data_
                && reinterpret_cast<char*>(scan_location)
                    < allocator->data_ + AllocatorT::kDataBufferSize);
            FieldType scan_value = *scan_location;

            if (scan_value == nullptr) return;
            // Check if value points to an object of type DefragT.
            if (scan_value->get_type() != BlockHelper<DefragT>::kIndex) return;

            // Calculate block index of scan_value.
            // TODO: Find a better way to do this.
            // TODO: There could be some random garbage in the field if it was
            // not initialized. Replace asserts with if-check and return stmt.
            char* block_base =
                PointerHelper::block_base_from_obj_ptr(scan_value);
            assert(block_base >= allocator->data_
                   && block_base < allocator->data_ + AllocatorT::kDataBufferSize);
            assert((block_base - allocator->data_) % AllocatorT::kBlockSizeBytes == 0);

#ifdef OPTION_DEFRAG_FORWARDING_POINTER
            auto* source_block = reinterpret_cast<typename BlockHelper<DefragT>::BlockType*>(block_base);
            if (source_block->has_forwarding) {
              // Rewrite this pointer.
              auto src_obj_id = PointerHelper::obj_id_from_obj_ptr(scan_value);
              bool valid_pointer = (source_block->allocation_bitmap() & (1ULL << src_obj_id)) != 0;

              if (valid_pointer) {
                auto* forwarding_ptr = reinterpret_cast<FieldType>(
                    source_block->get_forwarding_pointer(src_obj_id));
                assert(forwarding_ptr->get_type() == scan_value->get_type());

                *scan_location = forwarding_ptr;
              }
            }
#else
            auto scan_block_idx = (block_base - allocator->data_)
                / AllocatorT::kBlockSizeBytes;
            assert(scan_block_idx < AllocatorT::N);

            // Look for defrag record for this block.
            const auto record_id = block_idx_hash<NumRecords>(scan_block_idx);
            assert(record_id < NumRecords);
            const auto& record = records[record_id];

            if (record.source_block_idx == scan_block_idx) {
              // This pointer must be rewritten.
              auto src_obj_id = PointerHelper::obj_id_from_obj_ptr(scan_value);

              // ... but this pointer could contain garbage data.
              // In that case, we do not want need to rewrite.
#ifdef OPTION_DEFRAG_USE_GLOBAL
              bool valid_pointer =
                  allocator->defrag_records_.source_bitmap[record_id]
                  & (1ULL << src_obj_id);
#else
              bool valid_pointer = record.source_bitmap & (1ULL << src_obj_id);
#endif  // OPTION_DEFRAG_USE_GLOBAL
              if (!valid_pointer) return;

              // First src_obj_id bits are set to 1.
              BlockBitmapT cnt_mask = src_obj_id ==
                 63 ? (~0ULL) : ((1ULL << (src_obj_id + 1)) - 1);
              assert(__popcll(cnt_mask) == src_obj_id + 1);
#ifdef OPTION_DEFRAG_USE_GLOBAL
              int src_bit_idx = __popcll(cnt_mask
                  & allocator->defrag_records_.source_bitmap[record_id]) - 1;
#else
              int src_bit_idx = __popcll(cnt_mask & record.source_bitmap) - 1;
#endif  // OPTION_DEFRAG_USE_GLOBAL

              // Find correct target block and bit in target bitmaps.
              BlockBitmapT target_bitmap;
              auto target_block_idx = kInvalidBlockIndex;
              for (int i = 0; i < kDefragFactor; ++i) {
#ifdef OPTION_DEFRAG_USE_GLOBAL
                target_bitmap =
                    allocator->defrag_records_.target_bitmap[i][record_id];
#else
                target_bitmap = record.target_bitmap[i];
#endif  // OPTION_DEFRAG_USE_GLOBAL

                if (__popcll(target_bitmap) > src_bit_idx) {
                  // Target block found.
#ifdef OPTION_DEFRAG_USE_GLOBAL
                  target_block_idx =
                      allocator->defrag_records_.target_block_idx[i][record_id];
#else
                  target_block_idx = record.target_block_idx[i];
#endif  // OPTION_DEFRAG_USE_GLOBAL
                  break;
                } else {
                  src_bit_idx -= __popcll(target_bitmap);
                }
              }

              // Assert that target block was found.
              assert(target_block_idx < AllocatorT::N);

              // Find src_bit_idx-th bit in target bitmap.
              for (int j = 0; j < src_bit_idx; ++j) {
                target_bitmap &= target_bitmap - 1;
              }
              int target_obj_id = __ffsll(target_bitmap) - 1;
              assert(target_obj_id < BlockHelper<DefragT>::kSize);
              assert(target_obj_id >= 0);
              assert((allocator->template get_block<DefragT>(
                          target_block_idx)->free_bitmap
                      & (1ULL << target_obj_id)) == 0);

              // Rewrite pointer.
              auto* target_block = allocator->template get_block<
                  typename std::remove_pointer<FieldType>::type>(
                      target_block_idx);
              *scan_location = PointerHelper::rewrite_pointer(
                      scan_value, target_block, target_obj_id);

#ifndef NDEBUG
              // Sanity checks.
              assert(PointerHelper::block_base_from_obj_ptr(*scan_location)
                  == reinterpret_cast<char*>(target_block));
              assert((*scan_location)->get_type()
                  == BlockHelper<DefragT>::kIndex);
              auto* loc_block = reinterpret_cast<typename BlockHelper<DefragT>::BlockType*>(
                  PointerHelper::block_base_from_obj_ptr(*scan_location));
              assert(loc_block->type_id == BlockHelper<DefragT>::kIndex);
              assert((loc_block->free_bitmap & (1ULL << target_obj_id)) == 0);
#endif  // NDEBUG
            }
#endif  // OPTION_DEFRAG_FORWARDING_POINTER
          }
        };

        // Do not scan this field.
        template<int Dummy>
        struct FieldSelector<false, Dummy> {
          template<typename BlockT>
          __DEV__ static void call(AllocatorT* allocator,
                                   BlockT* block, ObjectIndexT object_id) {}
        };

        template<typename... Args>
        __DEV__ bool operator()(Args... args) {
          // Rewrite field if field type is a super class (or exact class)
          // of DefragT.
          FieldSelector<std::is_pointer<FieldType>::value
              && std::is_base_of<typename std::remove_pointer<FieldType>::type,
                                 DefragT>::value, 0>::call(args...);
          return true;  // Continue processing.
        }
      };

      bool operator()(AllocatorT* allocator, bool first_iteration) {
        static_assert(NumRecords <= kMaxDefragRecords,
                      "Too many defragmentation records requested.");

        bool process_class = SoaClassHelper<ScanClassT>::template for_all<
            FieldChecker, /*IterateBase=*/ true>();
        if (process_class) {
          // Initialized during first iteration.
          static BlockIndexT num_soa_blocks;

          if (first_iteration) {
            // Initialize iteration: Perform scan operation on bitmap.
            allocator->allocated_[kScanTypeIndex].scan();

            auto* d_num_soa_blocks_ptr =
                &allocator->allocated_[kScanTypeIndex]
                    .data_.enumeration_result_size;
            num_soa_blocks = copy_from_device(d_num_soa_blocks_ptr);
          }

          if (num_soa_blocks > 0) {
            auto total_threads = num_soa_blocks * kScanBlockSize;
            kernel_iterate_rewrite_objects<ThisClass>
                <<<(total_threads + kCudaBlockSize - 1)/kCudaBlockSize,
                  kCudaBlockSize,
#ifdef OPTION_DEFRAG_FORWARDING_POINTER
                  /* No shared memory */
                  0
#else
                  NumRecords*sizeof(DefragRecord<BlockBitmapT>)
#endif  // OPTION_DEFRAG_FORWARDING_POINTER
                  >>>(allocator);
            gpuErrchk(cudaDeviceSynchronize());
          }
        }

        return true;  // Continue processing.
      }
    };
  };

  template<typename T>
  struct SoaObjectCopier {
    // Copies a single field value from one block to another one.
    template<typename SoaFieldHelperT>
    struct ObjectCopyHelper {
      using SoaFieldType = SoaField<typename SoaFieldHelperT::OwnerClass,
                                    SoaFieldHelperT::kIndex>;

      __DEV__ bool operator()(char* source_block_base, char* target_block_base,
                              ObjectIndexT source_slot, ObjectIndexT target_slot) {
        assert(source_slot < BlockHelper<T>::kSize);
        assert(target_slot < BlockHelper<T>::kSize);

        // TODO: Optimize copy routine for single value. Should not use the
        // assignment operator here.
        typename SoaFieldHelperT::type* source_ptr =
            SoaFieldType::template data_ptr_from_location<BlockHelper<T>::kSize>(
                source_block_base, source_slot);
        typename SoaFieldHelperT::type* target_ptr =
            SoaFieldType::template data_ptr_from_location<BlockHelper<T>::kSize>(
                target_block_base, target_slot);

        *target_ptr = *source_ptr;

#ifndef NDEBUG
        // Reset value for debugging purposes.
        memset(source_ptr, 0, sizeof(typename SoaFieldHelperT::type));
#endif  // NDEBUG

        return true;  // Continue processing.
      }
    };
  };
};


// Select a source block for fragmentation. Will not bring the number of
// defragmentation candidates under min_remaining_records.
template<BlockIndexT N_Objects, class... Types>
template<typename T, int NumRecords>
__DEV__ void SoaAllocator<N_Objects, Types...>::defrag_choose_source_block(
    int min_remaining_records) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (; tid < NumRecords; tid += blockDim.x * gridDim.x) {
    auto bid = kInvalidBlockIndex;

    // Assuming 64-bit bitmaps.
    for (auto bit = tid; bit < kN; bit += NumRecords) {
      auto container = leq_50_[BlockHelper<T>::kIndex].get_container(bit/64);
      if (container & (1ULL << (bit % 64))) {
        assert(leq_50_[BlockHelper<T>::kIndex][bit]);
        bid = bit;
        assert(block_idx_hash<NumRecords>(bid) == tid);
        break;
      }
    }

    defrag_records_.source_block_idx[tid] = bid;

    if (bid != kInvalidBlockIndex) {
      // This block would be suitable. Check if we still need more blocks
      if (atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1)
          > min_remaining_records) {
        defrag_records_.source_bitmap[tid] =
            ~get_block<T>(bid)->free_bitmap
            & BlockHelper<T>::BlockType::kBitmapInitState;

        // Remove from leq_50 to avoid block from being selected as target.
        ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].deallocate<true>(bid));

#ifdef OPTION_DEFRAG_FORWARDING_POINTER
        get_block<T>(bid)->has_forwarding = true;
#endif  // OPTION_DEFRAG_FORWARDING_POINTER
      } else {
        atomicAdd(&num_leq_50_[BlockHelper<T>::kIndex], 1);
        defrag_records_.source_block_idx[tid] = kInvalidBlockIndex;
        break;
      }
    }
  }

  // We got enough blocks. Fill up the rest with invalid markers.
  for (tid += blockDim.x * gridDim.x; tid < NumRecords;
       tid += blockDim.x * gridDim.x) {
    defrag_records_.source_block_idx[tid] = kInvalidBlockIndex;
  }
}


// TODO: Allow a block to be a target multiple times.
template<BlockIndexT N_Objects, class... Types>
template<typename T, int NumRecords>
__DEV__ void SoaAllocator<N_Objects, Types...>::defrag_choose_target_blocks() {
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
       tid < NumRecords; tid += blockDim.x * gridDim.x) {
    if (defrag_records_.source_block_idx[tid] != kInvalidBlockIndex) {
      int remaining_slots = __popcll(defrag_records_.source_bitmap[tid]);

      // Find target blocks.
      for (int i = 0; i < kDefragFactor; ++i) {
        // Note: May have to turn on these bits again later.
        // But not for now, since block should not be chosen again.
        auto bid = leq_50_[BlockHelper<T>::kIndex].deallocate_seed(tid + i);
        defrag_records_.target_block_idx[i][tid] = bid;

        const auto target_bitmap = get_block<T>(bid)->free_bitmap;
        defrag_records_.target_bitmap[i][tid] = target_bitmap;
        remaining_slots -= __popcll(target_bitmap);

        if (remaining_slots <= 0) break;
      }

      assert(remaining_slots <= 0);
    }
  }
}


template<BlockIndexT N_Objects, class... Types>
template<typename T, int NumRecords>
__DEV__ void SoaAllocator<N_Objects, Types...>::defrag_move() {
  // Use 64 threads per SOA block.
  assert(blockDim.x % 64 == 0);

  for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
       tid < 64 * NumRecords; tid += blockDim.x * gridDim.x) {
    const int source_pos = tid % 64;
    const int record_id = tid / 64;
    const auto source_block_idx = defrag_records_.source_block_idx[record_id];

    if (source_block_idx != kInvalidBlockIndex) {
      BlockBitmapT source_bitmap = defrag_records_.source_bitmap[record_id];

      // This thread should move the source_pos-th object (if it exists).
      if (source_pos < __popcll(source_bitmap)) {
        // Determine positition in source bitmap: Find index of source_pos-th set bit.
        for (int i = 0; i < source_pos; ++i) {
          source_bitmap &= source_bitmap - 1;
        }
        int source_object_id = __ffsll(source_bitmap) - 1;
        assert(source_object_id >= 0);

        // Determine target block and target position.
        int target_pos = source_pos;
        auto target_block_idx = kInvalidBlockIndex;
        BlockBitmapT target_bitmap;

        for (int i = 0; i < kDefragFactor; ++i) {
          target_bitmap = defrag_records_.target_bitmap[i][record_id];
          const auto num_slots = __popcll(target_bitmap);
          if (target_pos < num_slots) {
            // This object goes in here.
            target_block_idx = defrag_records_.target_block_idx[i][record_id];
            break;
          } else {
            target_pos -= num_slots;
          }
        }
        assert(target_block_idx != kInvalidBlockIndex);

        // Determine target object ID: Find index of target_pos-th set bit.
        for (int i = 0; i < target_pos; ++i) {
          target_bitmap &= target_bitmap - 1;
        }
        int target_object_id = __ffsll(target_bitmap) - 1;
        assert(target_object_id >= 0);

        auto* source_block = get_block<T>(source_block_idx);
        auto* target_block = get_block<T>(target_block_idx);

        SoaClassHelper<T>::template dev_for_all<AllocatorWrapperDefrag<ThisAllocator>
            ::template SoaObjectCopier<T>::ObjectCopyHelper, true>(
                reinterpret_cast<char*>(source_block),
                reinterpret_cast<char*>(target_block),
                source_object_id, target_object_id);

#ifdef OPTION_DEFRAG_FORWARDING_POINTER
        // Store forwarding pointer.
        auto* target_obj_ptr = target_block->make_pointer(target_object_id);
        assert(target_obj_ptr->get_type() == BlockHelper<T>::kIndex);
        assert(target_obj_ptr->get_type() == source_block->get_static_type());
        source_block->set_forwarding_pointer(source_object_id, target_obj_ptr);
#endif  // OPTION_DEFRAG_FORWARDING_POINTER
      }
    }
  }
}


template<BlockIndexT N_Objects, class... Types>
template<typename T, int NumRecords>
__DEV__ void SoaAllocator<N_Objects, Types...>::defrag_update_block_state() {
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
       tid < NumRecords; tid += blockDim.x * gridDim.x) {
    if (defrag_records_.source_block_idx[tid] != kInvalidBlockIndex) {
      // Delete source block.
      // Invalidate block.
      get_block<T>(defrag_records_.source_block_idx[tid])->free_bitmap = 0ULL;
      // Precond.: Block is active and allocated. Block was already
      // removed from leq_50_ above.
      deallocate_block<T>(defrag_records_.source_block_idx[tid],
                          /*dealloc_leq_50=*/ false);

      // Update state of target blocks.
      int remaining_objs = __popcll(defrag_records_.source_bitmap[tid]);

      for (int i = 0; i < kDefragFactor; ++i) {
        auto target_bitmap = defrag_records_.target_bitmap[i][tid];
        const auto target_block_idx = defrag_records_.target_block_idx[i][tid];
        const auto num_target_slots = __popcll(target_bitmap);

        if (num_target_slots <= remaining_objs) {
          // This target block is now full.
          ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(target_block_idx));
          atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
          get_block<T>(target_block_idx)->free_bitmap = 0ULL;
          // leq_50 is already cleared.
        } else {
          // Check leq_50 status of target block.
          int num_target_after = BlockHelper<T>::kSize - num_target_slots + remaining_objs;
          if (num_target_after <= BlockHelper<T>::kLeq50Threshold) {
            // This block is still leq_50.
            ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].allocate<true>(target_block_idx));
          } else {
            atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
          }

          // Clear now allocated bits.
          for (int j = 0; j < remaining_objs; ++j) {
            target_bitmap &= target_bitmap - 1;
          }

          get_block<T>(target_block_idx)->free_bitmap = target_bitmap;
        }

        remaining_objs -= num_target_slots;
        if (remaining_objs <= 0) {
          // This is the last target block.
          break;
        }
      }

      assert(remaining_objs <= 0);
    }
  }
}


template<BlockIndexT N_Objects, class... Types>
template<int NumRecords>
__DEV__ void SoaAllocator<N_Objects, Types...>
    ::load_records_to_shared_mem() {
  extern __shared__ DefragRecord<BlockBitmapT> records[];

  // Every block loads records into shared memory.
  for (int i = threadIdx.x; i < NumRecords; i += blockDim.x) {
    records[i].copy_from(defrag_records_, i);
  }

  __syncthreads();
}


// For benchmarks: Measure time spent outside of parallel sections.
// Measure time in microseconds because numbers are small.
long unsigned int bench_init_leq_time = 0;
long unsigned int bench_defrag_move_time = 0;
long unsigned int bench_rewrite_time = 0;
long unsigned int bench_total_defrag_time = 0;

// Should be invoked from host side. Perform defragmentation only if there are
// enough blocks so that at least min_num_compactions many defragmentation
// candidates can be elimated. (Not taking into account collisions.)
template<BlockIndexT N_Objects, class... Types>
template<typename T, int NumRecords>
void SoaAllocator<N_Objects, Types...>::parallel_defrag(int min_num_compactions) {
  static_assert(NumRecords <= kMaxDefragRecords, "Too many records requested.");
  assert(min_num_compactions >= 1);

  auto time_0 = std::chrono::system_clock::now();

  // Determine number of records.
  auto num_leq_blocks =
      copy_from_device(&num_leq_50_[BlockHelper<T>::kIndex]);

  for (bool first_iteration = true; ; first_iteration = false) {
    // E.g.: n = 3: Retain 3/4 = 75% of blocks. Round up.
    const int max_num_source_blocks =
        (num_leq_blocks - kMinDefragRetainBlocks) / (kDefragFactor + 1);
    if (max_num_source_blocks <= min_num_compactions) break;

    const int min_remaining_records = num_leq_blocks - max_num_source_blocks;
    assert(min_remaining_records >= (num_leq_blocks - min_remaining_records)
           * kDefragFactor);

#ifndef NDEBUG
    printf("[DEFRAG]  min_remaining:  %i / %i\n",
           (int) min_remaining_records, (int) num_leq_blocks);
#endif  // NDEBUG

    auto time_1 = std::chrono::system_clock::now();

    // TODO: Assign one warp per defrag record.
    // Step 1: Choose source blocks.
    member_func_kernel<
        ThisAllocator, int,
        &ThisAllocator::template defrag_choose_source_block<T, NumRecords>>
        <<<512, (NumRecords + 512 - 1) / 512>>>(this, min_remaining_records);
    gpuErrchk(cudaDeviceSynchronize());

    // Step 2: Choose target blocks.
    member_func_kernel<
        ThisAllocator,
        &ThisAllocator::defrag_choose_target_blocks<T, NumRecords>>
        <<<512, (NumRecords + 512 - 1) / 512>>>(this);
    gpuErrchk(cudaDeviceSynchronize());

    auto time_2 = std::chrono::system_clock::now();
    bench_init_leq_time += std::chrono::duration_cast<std::chrono::microseconds>(
        time_2 - time_1).count();

    // Move objects. 64 threads per block.
    const int num_move_threads = 64*NumRecords;
    member_func_kernel<
        ThisAllocator,
        &ThisAllocator::defrag_move<T, NumRecords>>
        <<<(num_move_threads + 256 - 1) / 256, 256>>>(this);
    gpuErrchk(cudaDeviceSynchronize());

    // Update block states.
    member_func_kernel<
        ThisAllocator,
        &ThisAllocator::defrag_update_block_state<T, NumRecords>>
        <<<512, (NumRecords + 512 - 1) / 512>>>(this);
    gpuErrchk(cudaDeviceSynchronize());

    auto time_3 = std::chrono::system_clock::now();
    bench_defrag_move_time += std::chrono::duration_cast<std::chrono::microseconds>(
        time_3 - time_2).count();

    // Scan and rewrite pointers.
    TupleHelper<Types...>
        ::template for_all<AllocatorWrapperDefrag<ThisAllocator>
        ::template SoaPointerUpdater<T, NumRecords>
        ::template ClassIterator>(this, first_iteration);

    auto time_4 = std::chrono::system_clock::now();
    bench_rewrite_time += std::chrono::duration_cast<std::chrono::microseconds>(
        time_4 - time_3).count();

    // Determine new number of defragmentation candidates.
#ifndef NDEBUG
    auto num_leq_blocks_before = num_leq_blocks;
#endif  // NDEBUG

    num_leq_blocks = copy_from_device(&num_leq_50_[BlockHelper<T>::kIndex]);

#ifndef NDEBUG
    printf("[DEFRAG]  Completed pass<%i> [%s]:  %i --> %i   (%i)\n",
           NumRecords, typeid(T).name(),
           num_leq_blocks_before, num_leq_blocks, num_leq_blocks_before - num_leq_blocks);
#endif  // NDEBUG
  }

  auto time_5 = std::chrono::system_clock::now();
  bench_total_defrag_time += std::chrono::duration_cast<std::chrono::microseconds>(
      time_5 - time_0).count();
}

template<BlockIndexT N_Objects, class... Types>
template<typename T>
void SoaAllocator<N_Objects, Types...>::parallel_defrag(int min_num_compactions) {
  this->template parallel_defrag<T, kMaxDefragRecords>(min_num_compactions);
}

template<BlockIndexT N_Objects, class... Types>
void SoaAllocator<N_Objects, Types...>::DBG_print_defrag_time() {
  printf("%lu, %lu, %lu, %lu\n",
         bench_init_leq_time / 1000,
         bench_defrag_move_time / 1000,
         bench_rewrite_time / 1000,
         bench_total_defrag_time / 1000);
}
