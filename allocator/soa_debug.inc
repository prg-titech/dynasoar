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
      printf("number of fields(%s) = %lu\n", typeid(T).name(), sizeof(T));
      printf("block size(%s) = %i\n", typeid(T).name(), BlockHelper<T>::kSize);
      printf("leq50 threshold(%s) = %i\n", typeid(T).name(),
             BlockHelper<T>::kLeq50Threshold);
      printf("data segment bytes(%s) = %i\n", typeid(T).name(),
             BlockHelper<T>::kBytes);
      printf("block bytes(%s) = %lu\n", typeid(T).name(),
             sizeof(typename BlockHelper<T>::BlockType));
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
  printf("----------------------------------------------------------\n");
  TupleHelper<Types...>::template for_all<
      AllocatorWrapperDebug<ThisAllocator>::template SoaTypeDbgPrinter>();
  printf("Smallest block type: %s at %i bytes.\n",
         typeid(typename TupleHelper<Types...>::Type64BlockSizeMin).name(),
         TupleHelper<Types...>::kPadded64BlockMinSize);
  printf("Block size bytes: %i\n", kBlockSizeBytes);
  printf("Data buffer size (MB): %f\n", kDataBufferSize/1024.0/1024.0);
  printf("----------------------------------------------------------\n");
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
