// Textual header.

static const BlockIndexT kInvalidBlockIndex =
    std::numeric_limits<BlockIndexT>::max();

template<int NumBuckets, typename IndexT>
__DEV__ IndexT block_idx_hash(IndexT block_idx) {
  return (block_idx / 64) % NumBuckets;
}

template<typename AllocatorT>
struct AllocatorWrapperDefrag {
  template<typename T>
  using BlockHelper = typename AllocatorT::template BlockHelper<T>;

  using BlockBitmapT = typename AllocatorT::BlockBitmapT;


  // Select fields of type DefragT* and rewrite pointers if necessary.
  // DefragT: Type that was defragmented.
  // ScanClassT: Class which is being scanned for affected fields.
  // SoaFieldHelperT: SoaFieldHelper of potentially affected field.
  template<typename DefragT, int NumRecords>
  struct SoaPointerUpdater {
    template<typename ScanClassT>
    struct ClassIterator {
      using ThisClass = ClassIterator<ScanClassT>;

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

      // Scan and rewrite field.
      template<typename SoaFieldHelperT>
      struct FieldUpdater {
        using FieldType = typename SoaFieldHelperT::type;

        using SoaFieldType = SoaField<typename SoaFieldHelperT::OwnerClass,
                                      SoaFieldHelperT::kIndex>;

        // Scan this field.
        template<bool Check, int Dummy> 
        struct FieldSelector {
          template<typename... Args>
          __DEV__ static void call(AllocatorT* allocator,
                                   ScanClassT* object) {
            extern __shared__ DefragRecord<BlockBitmapT> records[];

            // Location of field value to be scanned/rewritten.
            // TODO: This is inefficient. We first build a pointer and then
            // disect it again. (Inside *_from_obj_ptr().)
            FieldType* scan_location =
                SoaFieldType::data_ptr_from_obj_ptr(object);
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
            auto scan_block_idx = (block_base - allocator->data_)
                / AllocatorT::kBlockSizeBytes;
            assert(scan_block_idx < AllocatorT::N);

            // Look for defrag record for this block.
            const auto record_id = block_idx_hash<NumRecords>(scan_block_idx);
            assert(record_id < NumRecords);
            const auto& record = records[record_id];

            if (record.source_block_idx == scan_block_idx) {
              // This pointer must be rewritten.
              int src_obj_id = PointerHelper::obj_id_from_obj_ptr(scan_value);

              // ... but this pointer could contain garbage data.
              // In that case, we do not want need to rewrite.
              bool valid_pointer = record.source_bitmap & (1ULL << src_obj_id);
              if (!valid_pointer) return;

              // First src_obj_id bits are set to 1.
              BlockBitmapT cnt_mask = src_obj_id ==
                 63 ? (~0ULL) : ((1ULL << (src_obj_id + 1)) - 1);
              assert(__popcll(cnt_mask) == src_obj_id + 1);
              int src_bit_idx = __popcll(cnt_mask & record.source_bitmap) - 1;

              // Find correct target block and bit in target bitmaps.
              BlockBitmapT target_bitmap;
              auto target_block_idx = kInvalidBlockIndex;
              for (int i = 0; i < kDefragFactor; ++i) {
                target_bitmap = record.target_bitmap[i];

                if (__popcll(target_bitmap) > src_bit_idx) {
                  // Target block found.
                  target_block_idx = record.target_block_idx[i];
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
          }
        };

        // Do not scan this field.
        template<int Dummy>
        struct FieldSelector<false, Dummy> {
          __DEV__ static void call(AllocatorT* allocator,
                                   ScanClassT* object) {}
        };

        __DEV__ bool operator()(AllocatorT* allocator, ScanClassT* object) {
          // Rewrite field if field type is a super class (or exact class)
          // of DefragT.
          FieldSelector<std::is_pointer<FieldType>::value
              && std::is_base_of<typename std::remove_pointer<FieldType>::type,
                                 DefragT>::value, 0>::call(
              allocator, object);
          return true;  // Continue processing.
        }
      };

      bool operator()(AllocatorT* allocator) {
        static_assert(NumRecords <= kMaxDefragRecords,
                      "Too many defragmentation records requested.");

        bool process_class = SoaClassHelper<ScanClassT>::template for_all<
            FieldChecker, /*IterateBase=*/ true>();
        if (process_class) {
          ParallelExecutor<AllocatorT, ScanClassT, ScanClassT, void,
                           SoaBase<AllocatorT>, /*Args...=*/ AllocatorT*>
              ::template FunctionWrapper<&SoaBase<AllocatorT>
                  ::template rewrite_object<ThisClass, ScanClassT>>
              ::template WithPre<&AllocatorT::template load_records_to_shared_mem<NumRecords>>
              ::parallel_do(allocator,
                            NumRecords*sizeof(DefragRecord<BlockBitmapT>),
                            allocator);
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
                              uint8_t source_slot, uint8_t target_slot) {
        assert(source_slot < BlockHelper<T>::kSize);
        assert(target_slot < BlockHelper<T>::kSize);

        // TODO: Optimize copy routine for single value. Should not use the
        // assignment operator here.
        // TODO: Make block size a template parameter.
        typename SoaFieldHelperT::type* source_ptr =
            SoaFieldType::data_ptr_from_location(
                source_block_base, BlockHelper<T>::kSize, source_slot);
        typename SoaFieldHelperT::type* target_ptr =
            SoaFieldType::data_ptr_from_location(
                target_block_base, BlockHelper<T>::kSize, target_slot);

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
    for (auto cid = tid; cid < kN/64; cid += NumRecords) {
      auto container = leq_50_[BlockHelper<T>::kIndex].get_container(cid);
      if (container != 0ULL) {
        bid = 64*cid + __ffsll(container) - 1;
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
      } else {
        atomicAdd(&num_leq_50_[BlockHelper<T>::kIndex], 1);
        break;
      }
    }
  }

  // We got enough blocks. Fill up the rest with invalid markers.
  for (; tid < NumRecords; tid += blockDim.x * gridDim.x) {
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

        SoaClassHelper<T>::template dev_for_all<AllocatorWrapperDefrag<ThisAllocator>
            ::template SoaObjectCopier<T>::ObjectCopyHelper, true>(
                reinterpret_cast<char*>(get_block<T>(source_block_idx)),
                reinterpret_cast<char*>(get_block<T>(target_block_idx)),
                source_object_id, target_object_id);
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
    ::load_records_to_shared_mem(SoaAllocator<N_Objects, Types...>* allocator) {
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

// Should be invoked from host side.
template<BlockIndexT N_Objects, class... Types>
template<typename T, int NumRecords>
void SoaAllocator<N_Objects, Types...>::parallel_defrag(int min_leq_blocks) {
  assert(NumRecords <= kMaxDefragRecords);

  // Retain at least min_leq_blocks many blocks.
  assert(min_leq_blocks >= 0);

  // Determine number of records.
  auto num_leq_blocks =
      copy_from_device(&num_leq_50_[BlockHelper<T>::kIndex]);
  auto num_records = num_leq_blocks - min_leq_blocks;

  while (num_records > min_leq_blocks) {
    // E.g.: n = 3: Retain 3/4 = 75% of blocks. Round up.
    const int min_remaining_records =
        ((num_records + 1) * kDefragFactor) / (kDefragFactor + 1);
    assert(min_remaining_records
           >= num_records * kDefragFactor / (kDefragFactor + 1));

    if (min_remaining_records >= num_leq_blocks) {
      // Nothing to do.
      break;
    }

    auto time_1 = std::chrono::system_clock::now();

    // TODO: Assign one warp per defrag record.
    // Step 1: Choose source blocks.
    member_func_kernel_1<
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
        ::template ClassIterator>(this);

    auto time_4 = std::chrono::system_clock::now();
    bench_rewrite_time += std::chrono::duration_cast<std::chrono::microseconds>(
        time_4 - time_3).count();

    // Determine new number of defragmentation candidates.
#ifndef NDEBUG
    auto num_leq_blocks_before = num_leq_blocks;
#endif  // NDEBUG

    num_leq_blocks = copy_from_device(&num_leq_50_[BlockHelper<T>::kIndex]);
    num_records = num_leq_blocks - min_leq_blocks;

#ifndef NDEBUG
    printf("Completed defragmention pass [%s]:  %i --> %i   (%i)\n", typeid(T).name(),
           num_leq_blocks_before, num_leq_blocks, num_leq_blocks_before - num_leq_blocks);
#endif  // NDEBUG
  }
}

template<BlockIndexT N_Objects, class... Types>
void SoaAllocator<N_Objects, Types...>::DBG_print_defrag_time() {
  printf("%lu, %lu, %lu\n",
         bench_init_leq_time / 1000,
         bench_defrag_move_time / 1000,
         bench_rewrite_time / 1000);
}
