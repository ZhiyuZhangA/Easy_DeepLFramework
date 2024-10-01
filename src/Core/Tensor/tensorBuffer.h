#ifndef _TENSOR_BUFFER_
#define _TENSOR_BUFFER_

#include <cstdint>
#include <memory>
#include "../allocators/default_allocator.h"

namespace NN {
    template <typename T>
    class tensorBuffer {
    public:
        // The byte_size refers to the total amount of bytes that will be allocated by the machine.
        explicit tensorBuffer(size_t byte_size, void* ptr=nullptr) : _data(ptr), _byte_size(byte_size)  {
            if (!ptr) {
                _data = _alloc.allocate(_byte_size);
            }
            _data_t = nullptr;
        }

        ~tensorBuffer() {
            // if (_data_t != nullptr) _data_t.~T();
            _alloc.deallocate(_data, _byte_size);
            _data_t = nullptr;
        }

        // Returns the raw memory in the buffer
        inline void*& rdata() { return _data; }

        // Returns the number of bytes allocated for the raw memory.
        inline size_t byte_size() const { return _byte_size; }

        // Returns the memory of data type in the buffer
        inline T*& data_ptr_t() { return _data_t; }

    private:
        void* _data;
        T* _data_t;
        size_t _byte_size;
        d_alloc _alloc;
    };
}


#endif
