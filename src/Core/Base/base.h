#ifndef _BASE_
#define _BASE_

#include <cstdint>
#include <unordered_map>
#include <string>
#include <fstream>
#include <iostream>

namespace NN {

//    struct Dtype {
//        std::string type_name;
//        size_t hash_code;
//        size_t bytes;
//    };

    extern std::unordered_map<std::string, size_t> supported_types;

    // Returns number of bytes for each data type.
    enum class Dtype : size_t {
        float32 = 4,
        float64 = 8,
        int16 = 2,
        int32 = 4,
        int64 = 8,
        uint16 = 2,
        uint32 = 4,
        uint64 = 4,
    };

    typedef float     float32;
    typedef double    float64;
    typedef int16_t   int16;
    typedef int32_t   int32;
    typedef int64_t   int64;
    typedef uint16_t  uint16;
    typedef uint32_t  uint32;
    typedef uint64_t  uint64;

    void initializeNNEngine();

}


#endif
