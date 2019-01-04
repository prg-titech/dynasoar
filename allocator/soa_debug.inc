// Textual header.

template<typename AllocatorT>
struct AllocatorWrapperDebug {
  template<typename T>
  using BlockHelper = typename AllocatorT::template BlockHelper<T>;

  using BlockBitmapT = typename AllocatorT::BlockBitmapT;

  // Print compile-type configuration statistics about this allocator.
  template<typename T>
  struct SoaTypeDbgPrinter {
    bool operator()() {
      printf("┌───────────────────────────────────────────────────────────────────────┐\n");
      printf("│ Block stats for %37s (type ID %5i) │\n",
             typeid(T).name(), BlockHelper<T>::kIndex);
      printf("├────────────────────┬──────────────────────────────────────────────────┤\n");
      printf("│ #fields            │ %8i                                         │\n",
             (int) sizeof(T));
      printf("│ #objects / block   │ %8i                                         │\n",
             (int) BlockHelper<T>::kSize);
      printf("│ leq50 threshold    │ %8i                                         │\n",
             (int) BlockHelper<T>::kLeq50Threshold);
      printf("│ block size         │ %8i bytes                                   │\n",
             (int) sizeof(typename BlockHelper<T>::BlockType));
      printf("│ base class         │ %48s │\n",
              typeid(typename T::BaseClass).name());
      printf("│ is abstract        │ %8i                                         │\n",
             (int) T::kIsAbstract);
      printf("│ data seg. [%2i] sz  │ %8i bytes                                   │\n",
             (int) BlockHelper<T>::kSize, (int) BlockHelper<T>::kBytes);
      printf("│         (unpadded) │ %8i bytes                                   │\n",
             (int) SoaClassHelper<T>::template BlockConfig<BlockHelper<T>::kSize>
                 ::kDataSegmentSizeUnpadded);

      // Compute waste due to padding of data segment and within data segment.
      int simple_ds_size =
          SoaClassHelper<T>::template BlockConfig<BlockHelper<T>::kSize>
              ::kSimpleSize;
      printf("│        (simple sz) │ %8i bytes                                   │\n",
             simple_ds_size);
      printf("│    (padding waste) │ %8i bytes                                   │\n",
             (int) BlockHelper<T>::kBytes - simple_ds_size);

      // More stats...
      SoaClassHelper<T>::DBG_print_stats();
      return true;  // true means "continue processing".
    }
  };

  // Print runtime block usage statistics about this allocator.
  template<typename T>
  struct SoaTypeBlockDbgPrinter {
    __DEV__ bool operator()(AllocatorT* allocator, int* total_num_blk_alloc,
                            int* total_num_blk_leq50, int* total_num_blk_active,
                            int* total_num_obj_alloc, int* total_num_obj_used) {
      int num_blk_alloc = allocator->allocated_[BlockHelper<T>::kIndex]
          .DBG_count_num_ones();
      *total_num_blk_alloc += num_blk_alloc;

      int num_blk_leq50 = allocator->leq_50_[BlockHelper<T>::kIndex]
          .DBG_count_num_ones();
      *total_num_blk_leq50 += num_blk_leq50;

      int num_blk_active = allocator->active_[BlockHelper<T>::kIndex]
          .DBG_count_num_ones();
      *total_num_blk_active += num_blk_active;

      int num_obj_alloc = 0;
      int num_obj_used = 0;

      for (int i = 0; i < AllocatorT::kN; ++i) {
        if (allocator->allocated_[BlockHelper<T>::kIndex][i]) {
          auto* block = allocator->template get_block<T>(i);
          num_obj_alloc += BlockHelper<T>::kSize;
          num_obj_used += block->DBG_allocated_bits();
        }
      }

      *total_num_obj_alloc += num_obj_alloc;
      *total_num_obj_used += num_obj_used;
      float obj_frag = 1 - static_cast<float>(num_obj_used) / num_obj_alloc;

      printf("│ %2i │ %8i │ %8i │ %8i ││ %8i │ %8i │ %.6f │\n",
             BlockHelper<T>::kIndex, num_blk_alloc, num_blk_leq50, num_blk_active,
             num_obj_alloc, num_obj_used, obj_frag);
      return true;
    }
  };
};

// The number of allocated slots of a type. (#blocks * blocksize)
template<uint32_t N_Objects, class... Types>
template<class T>
__DEV__ uint32_t SoaAllocator<N_Objects, Types...>::DBG_allocated_slots() {
  uint32_t counter = 0;
  for (int i = 0; i < N; ++i) {
    if (allocated_[BlockHelper<T>::kIndex][i]) {
      counter += get_block<T>(i)->DBG_num_bits();
    }
  }
  return counter;
}

// The number of actually used slots of a type. (#blocks * blocksize)
template<uint32_t N_Objects, class... Types>
template<class T>
__DEV__ uint32_t SoaAllocator<N_Objects, Types...>::DBG_used_slots() {
  uint32_t counter = 0;
  for (int i = 0; i < N; ++i) {
    if (allocated_[BlockHelper<T>::kIndex][i]) {
      counter += get_block<T>(i)->DBG_allocated_bits();
    }
  }
  return counter;
}

template<uint32_t N_Objects, class... Types>
void SoaAllocator<N_Objects, Types...>::DBG_print_stats() {
  printf("┌───────────────────────────────────────────────────────────────────────┐\n");
  printf("│ Smallest block type:  %47s │\n",
         typeid(typename TupleHelper<Types...>::Type64BlockSizeMin).name());
  printf("│ Max. #objects:        %11i                                     │\n",
         (int) N_Objects);
  printf("│ Block size:           %11i bytes                               │\n", kBlockSizeBytes);
  printf("│ #Blocks:              %11i                                     │\n", N);
  printf("│ #Bitmap levels:       %11i                                     │\n",
          decltype(global_free_)::kLevels);
  printf("│ Data buffer size:      %6.6f MB                                  │\n", kDataBufferSize/1024.0/1024.0);
  printf("│ Allocator overead:    %11i bytes + block overhead              │\n",
         (int) sizeof(ThisAllocator));
  printf("│ Total memory usage:   %6.6f MB                                   │\n",
         (sizeof(ThisAllocator) + ThisAllocator::kDataBufferSize)/1024.0f/1024.0f);
  printf("└───────────────────────────────────────────────────────────────────────┘\n");

  TupleHelper<Types...>::template for_all<
      AllocatorWrapperDebug<ThisAllocator>::template SoaTypeDbgPrinter>();
}

template<uint32_t N_Objects, class... Types>
__DEV__ void SoaAllocator<N_Objects, Types...>::DBG_print_state_stats() {
  int num_blk_free = global_free_.DBG_count_num_ones();

  printf("┌────┬──────────┬──────────┬──────────┬┬──────────┬──────────┬──────────┐\n");
  printf("│ Ty │ #B_alloc │ #B_leq50 │ #B_activ ││ #O_alloc │  #O_used │   O_frag │\n");
  printf("├────┼──────────┼──────────┼──────────┼┼──────────┼──────────┼──────────┤\n");
  printf("│ fr │ %8i │      n/a │      n/a ││      n/a │      n/a │      n/a │\n",
         num_blk_free);

  // Accumulators for statistics.
  int total_num_blk_alloc = 0, total_num_blk_leq50 = 0;
  int total_num_blk_active = 0, total_num_obj_alloc = 0;
  int total_num_obj_used = 0;

  TupleHelper<Types...>::template dev_for_all<
      AllocatorWrapperDebug<ThisAllocator>::SoaTypeBlockDbgPrinter>(
          this, &total_num_blk_alloc, &total_num_blk_leq50,
          &total_num_blk_active, &total_num_obj_alloc, &total_num_obj_used);
  float total_obj_frag = 1 - static_cast<float>(total_num_obj_used)
      / total_num_obj_alloc;
  printf("│  Σ │ %8i │ %8i │ %8i ││ %8i │ %8i │ %.6f │\n",
         total_num_blk_alloc, total_num_blk_leq50, total_num_blk_active,
         total_num_obj_alloc, total_num_obj_used, total_obj_frag);
  printf("└────┴──────────┴──────────┴──────────┴┴──────────┴──────────┴──────────┘\n");
}

// Helper data structure for checking if a pointer (encoded type ID) belongs
// to BaseClass or one of its subclasses.
template<typename ThisAllocator, typename BaseClass>
struct PointerTypeChecker {
  // Iterating over all types T in the allocator.
  template<typename T>
  struct InnerHelper {
    // T is a subclass of BaseClass. Check if same type.
    template<bool Check, int Dummy>
    struct ClassSelector {
      __DEV__ static bool call(const BaseClass* obj) {
        if (obj->get_type() == ThisAllocator::template BlockHelper<T>::kIndex) {
          return false;  // No need to check other types.
        } else {
          return true;   // true means "continue processing".
        }
      }
    };

    // T is not a subclass of BaseClass. Skip.
    template<int Dummy>
    struct ClassSelector<false, Dummy> {
      __DEV__ static bool call(const BaseClass* obj) {
        return true;
      }
    };

    __DEV__ bool operator()(const BaseClass* obj) {
      return ClassSelector<std::is_base_of<BaseClass, T>::value, 0>::call(obj);
    }
  };
};

// Check type of pointer.
template<uint32_t N_Objects, class... Types>
template<class T>
__DEV__ bool SoaAllocator<N_Objects, Types...>::is_type(const T* ptr) {
  return TupleHelper<Types...>::template dev_for_all<PointerTypeChecker<
      ThisAllocator, T>::InnerHelper>(ptr);
}
