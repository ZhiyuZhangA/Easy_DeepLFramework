#include "default_allocator.h"
#include "immintrin.h"

NN::default_allocator::void_pointer NN::default_allocator::allocate(size_t size) {
    void_pointer result = _mm_malloc(size, 32);
    return result;
}

void NN::default_allocator::deallocate(void *p, size_t) {
    _mm_free(p);
}

NN::default_allocator::void_pointer NN::default_allocator::reallocate(void *p, size_t, size_t new_sz) {
    void_pointer result = realloc(p, new_sz);
    return result;
}

