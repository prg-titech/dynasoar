// Textual header.

static const unsigned int kDefragIndexEmpty =
    std::numeric_limits<unsigned int>::max();
static const unsigned int kDefragIndexBlocked =
    std::numeric_limits<unsigned int>::max() - 1;

__DEV__ int block_idx_hash(uint32_t block_idx, int num_buckets) {
  return block_idx % num_buckets;
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
  template<typename DefragT>
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
                                   ScanClassT* object, int num_records) {
            assert(num_records == allocator->num_defrag_records_);
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
            uint32_t scan_block_idx = (block_base - allocator->data_)
                / AllocatorT::kBlockSizeBytes;
            assert(scan_block_idx < AllocatorT::N);

            // Look for defrag record for this block.
            int record_id = block_idx_hash(scan_block_idx, num_records);
            assert(record_id < num_records);
            if (records[record_id].source_block_idx == scan_block_idx) {
              //printf("FOUND: %i at %i\n", (int) scan_block_idx, record_id);
              // This pointer must be rewritten.
              int src_obj_id = PointerHelper::obj_id_from_obj_ptr(scan_value);
              assert(src_obj_id < BlockHelper<DefragT>::kSize);
              assert((records[record_id].source_bitmap & (1ULL << src_obj_id)) != 0);

              // First src_obj_id bits are set to 1.
              BlockBitmapT cnt_mask = src_obj_id ==
                 63 ? (~0ULL) : ((1ULL << (src_obj_id + 1)) - 1);
              assert(__popcll(cnt_mask) == src_obj_id + 1);
              int src_bit_cnt =
                  __popcll(cnt_mask & records[record_id].source_bitmap) - 1;

              // Find src_bit_cnt-th bit in target bitmap.
              BlockBitmapT target_bitmap = records[record_id].target_bitmap;
              for (int j = 0; j < src_bit_cnt; ++j) {
                target_bitmap &= target_bitmap - 1;
              }
              int target_obj_id = __ffsll(target_bitmap) - 1;
              assert(target_obj_id < BlockHelper<DefragT>::kSize);
              assert(target_obj_id >= 0);
              assert((allocator->template get_block<DefragT>(
                      records[record_id].target_block_idx)->free_bitmap
                  & (1ULL << target_obj_id)) == 0);

              // Rewrite pointer.
              assert(records[record_id].target_block_idx < AllocatorT::N);
              auto* target_block = allocator->template get_block<
                  typename std::remove_pointer<FieldType>::type>(
                      records[record_id].target_block_idx);
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
                                   ScanClassT* object,
                                   int num_records) {}
        };

        __DEV__ bool operator()(AllocatorT* allocator, ScanClassT* object,
                                int num_records) {
          // Rewrite field if field type is a super class (or exact class)
          // of DefragT.
          FieldSelector<std::is_pointer<FieldType>::value
              && std::is_base_of<typename std::remove_pointer<FieldType>::type,
                                 DefragT>::value, 0>::call(
              allocator, object, num_records);
          return true;  // Continue processing.
        }
      };

      bool operator()(AllocatorT* allocator, int num_records) {
        bool process_class = SoaClassHelper<ScanClassT>::template for_all<
            FieldChecker, /*IterateBase=*/ true>();
        if (process_class) {
          // TODO: Optimize. No need to do another scan here.
          kernel_init_iteration<AllocatorT, ScanClassT><<<128, 128>>>(allocator);
          gpuErrchk(cudaDeviceSynchronize());

          ParallelExecutor<AllocatorT, ScanClassT, void,
                           SoaBase<AllocatorT>, /*Args...=*/ AllocatorT*, int>
              ::template FunctionWrapper<&SoaBase<AllocatorT>
                  ::template rewrite_object<ThisClass, ScanClassT>>
              ::template WithPre<&AllocatorT::load_records_to_shared_mem>
              ::parallel_do(allocator,
                            num_records*sizeof(DefragRecord<BlockBitmapT>),
                            allocator, num_records);
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


template<uint32_t N_Objects, class... Types>
template<typename T>
__DEV__ void SoaAllocator<N_Objects, Types...>::defrag_move(int num_records) {
  // Use 32 threads per SOA block, so that we can use warp shuffles instead
  // of shared memory.
  assert(blockDim.x % 32 == 0);
  int slot_id = threadIdx.x % 32;
  int record_id = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
  //int num_buckets = blockDim.x * gridDim.x / 32;
  const unsigned active = __activemask();

  // Problem: Cannot keep collision array in shared memory.

  // Defrag record.
  uint32_t source_block_idx, target_block_idx;
  BlockBitmapT source_bitmap, target_bitmap;

  if (slot_id == 0) {
    for (int i = 0; i < 3; ++i) {  // 3 tries
      source_block_idx = leq_50_[BlockHelper<T>::kIndex].deallocate_seed(record_id + i);
      assert(source_block_idx != (Bitmap<uint32_t, N>::kIndexError));

      record_id = block_idx_hash(source_block_idx, num_records);
      assert(record_id < num_records);
      unsigned int before = atomicCAS(&defrag_records_[record_id].source_block_idx,
                                      kDefragIndexEmpty,
                                      source_block_idx);

      if (before == kDefragIndexEmpty) {
        assert(defrag_records_[record_id].source_block_idx == source_block_idx);
        break;
      } else {
        // Collision detected.
        assert(defrag_records_[record_id].source_block_idx != source_block_idx
            && defrag_records_[record_id].source_block_idx != kDefragIndexEmpty);
        ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].allocate<true>(source_block_idx));
        source_block_idx = kDefragIndexEmpty;
      }
    }

    if (source_block_idx != kDefragIndexEmpty) {
      target_block_idx = leq_50_[BlockHelper<T>::kIndex].deallocate_seed(record_id + 509);
      assert(target_block_idx != (Bitmap<uint32_t, N>::kIndexError));

      // Block target block's slot.
      // TODO: Not ideal. It reduces the number of defrag records.
      int target_hash = block_idx_hash(target_block_idx, num_records);
      atomicCAS(&defrag_records_[target_hash].source_block_idx,
                kDefragIndexEmpty,
                kDefragIndexBlocked);

      // Invert free_bitmap to get a bitmap of allocations.
      source_bitmap = ~get_block<T>(source_block_idx)->free_bitmap
                      & BlockHelper<T>::BlockType::kBitmapInitState;
      assert(__popcll(source_bitmap) == BlockHelper<T>::kSize   // == N - #free
          - __popcll(get_block<T>(source_block_idx)->free_bitmap));
      // Target bitmap contains all free slots in target block.
      // TODO: Is is necessary to use atomicOr here?
      target_bitmap = atomicOr(&get_block<T>(target_block_idx)->free_bitmap, 0ULL);

      // Copy to global memory for scan step.
      defrag_records_[record_id].target_block_idx = target_block_idx;
      defrag_records_[record_id].source_bitmap = source_bitmap;
      defrag_records_[record_id].target_bitmap = target_bitmap;

      assert(__popcll(target_bitmap) >= __popcll(source_bitmap));
    }
  }

  // Shuffle defrag records from thread 0.
  source_block_idx = __shfl_sync(active, source_block_idx, 0);
  if (source_block_idx != kDefragIndexEmpty) {
    target_block_idx = __shfl_sync(active, target_block_idx, 0);
    source_bitmap = __shfl_sync(active, source_bitmap, 0);
    target_bitmap = __shfl_sync(active, target_bitmap, 0);

    // Find index of bit in bitmaps.
    int num_moves = __popcll(source_bitmap);
    assert(num_moves <= BlockHelper<T>::kSize/2);
    for (int i = 0; i < slot_id; ++i) {
      // Clear least significant bit.
      source_bitmap &= source_bitmap - 1;
      target_bitmap &= target_bitmap - 1;
    }
    int source_object_id = __ffsll(source_bitmap) - 1;
    int target_object_id = __ffsll(target_bitmap) - 1;

    // Move objects.
    if (source_object_id > -1) {
      assert(target_object_id > -1);

      SoaClassHelper<T>::template dev_for_all<AllocatorWrapperDefrag<ThisAllocator>
          ::template SoaObjectCopier<T>::ObjectCopyHelper, true>(
              reinterpret_cast<char*>(get_block<T>(source_block_idx)),
              reinterpret_cast<char*>(get_block<T>(target_block_idx)),
              source_object_id, target_object_id);
    }
    
    // Last thread performs all update, because it has access to the new
    // target_bitmap (with all slots occupied).
    if (slot_id == num_moves - 1) {
      // Invalidate source block.
      get_block<T>(source_block_idx)->free_bitmap = 0ULL;
      // Delete source block.
      // Precond.: Block is active and allocated. Block was already
      // removed from leq_50_ above.
      deallocate_block<T>(source_block_idx, /*dealloc_leq_50=*/ false);

      // Update free_bitmap in target block.
      target_bitmap &= target_bitmap - 1;   // Clear one more bit.
      //get_block<T>(target_block_idx)->free_bitmap = target_bitmap;
      atomicExch(&get_block<T>(target_block_idx)->free_bitmap, target_bitmap);
      // Make sure updated bitmap is visible to all threads before
      // (potentially) putting the block back into the leq_50_ bitmap.
      // Problem: Thread fence is not enough. There is no guarantee that
      // previous update is visible.
      __threadfence();

      // Update state of target block.
      int num_target_after = __popcll(
          ~target_bitmap & BlockHelper<T>::BlockType::kBitmapInitState);

      if (num_target_after == BlockHelper<T>::kSize) {
        // Block is now full.
         ASSERT_SUCCESS(active_[BlockHelper<T>::kIndex].deallocate<true>(
            target_block_idx));
      }

      if (num_target_after <= BlockHelper<T>::kLeq50Threshold) {
        // Block is still less than 50% full.
        ASSERT_SUCCESS(leq_50_[BlockHelper<T>::kIndex].allocate<true>(
            target_block_idx));
        atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 1);
      } else {
        // Deferred update from above. (2 block removed from bitmap.)
        atomicSub(&num_leq_50_[BlockHelper<T>::kIndex], 2);
      }
    }
  }
}


template<uint32_t N_Objects, class... Types>
__DEV__ void SoaAllocator<N_Objects, Types...>
    ::load_records_to_shared_mem(SoaAllocator<N_Objects, Types...>* allocator,
                                 int num_records) {
  extern __shared__ DefragRecord<BlockBitmapT> records[];
  for (int i = threadIdx.x; i < num_defrag_records_; i += blockDim.x) {
    records[i] = defrag_records_[i];
  }

  __syncthreads();
}


// Should be invoked from host side.
template<uint32_t N_Objects, class... Types>
template<typename T>
void SoaAllocator<N_Objects, Types...>::parallel_defrag(int max_records,
                                                        int min_records) {
  // Determine number of records.
  auto num_leq_blocks =
      copy_from_device(&num_leq_50_[BlockHelper<T>::kIndex]);
  int num_records = min(max_records, num_leq_blocks/2);
  num_records = max(0, num_records);

  if (num_records >= min_records) {
    // Create working copy of leq_50_ bitmap.
    kernel_initialize_leq<T><<<256, 256>>>(this, num_records);
    gpuErrchk(cudaDeviceSynchronize());

    // Move objects. 4 SOA block per CUDA block. 32 threads per block.
    // (Because blocks are at most 50% full.)
    kernel_defrag_move<T><<<
        (num_records + 4 - 1) / 4, 128>>>(this, num_records);
    gpuErrchk(cudaDeviceSynchronize());

    // Scan and rewrite pointers.
    TupleHelper<Types...>
        ::template for_all<AllocatorWrapperDefrag<ThisAllocator>
        ::template SoaPointerUpdater<T>::template ClassIterator>(
            this, num_records);
  }
}

template<uint32_t N_Objects, class... Types>
__DEV__ void SoaAllocator<N_Objects, Types...>::initialize_leq_collisions() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  defrag_records_[tid].source_block_idx = kDefragIndexEmpty;
}