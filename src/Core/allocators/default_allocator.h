//
// Created by zzy on 2024/7/16.
//

#ifndef _DEFAULT_ALLOCATOR_
#define _DEFAULT_ALLOCATOR_

#include "../Base/base.h"
#include <new>
#include <cstddef>
#include <cstdlib>
#include <climits>
#include <iostream>
#include <memory>

/*
 * TODO: 处理内存不足的情况
 * */
namespace NN {
    // Responsible for the uninitialized memory allocation and deallocation

    class default_allocator  {
    public:
        typedef void* void_pointer;

        default_allocator() = default;

        void_pointer allocate(size_t size);
        void deallocate(void *p, size_t);
        void_pointer reallocate(void *p, size_t, size_t new_sz);
    };

    typedef default_allocator d_alloc;
}



#endif
