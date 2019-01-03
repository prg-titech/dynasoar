#ifndef EXAMPLE_CONFIGURATION_SOA_ALLOC_ALLOCATOR_CONFIG_H
#define EXAMPLE_CONFIGURATION_SOA_ALLOC_ALLOCATOR_CONFIG_H

#ifdef CHK_ALLOCATOR_DEFINED
#error Allocator already defined
#else
#define CHK_ALLOCATOR_DEFINED
#endif  // CHK_ALLOCATOR_DEFINED

#include "allocator/soa_allocator.h"
#include "allocator/soa_base.h"
#include "allocator/allocator_handle.h"

#endif  // EXAMPLE_CONFIGURATION_SOA_ALLOC_ALLOCATOR_CONFIG_H
