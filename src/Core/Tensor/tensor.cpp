#include "tensor.h"

namespace NN {
    // Return the shape that the tensor would broadcast to
    std::vector<size_t> broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
        std::vector<size_t> result;
        auto it1 = shape1.rbegin();
        auto it2 = shape2.rbegin();

        while (it1 != shape1.rend() || it2 != shape2.rend()) {
            int dim1 = (it1 != shape1.rend()) ? *it1++ : 1;
            int dim2 = (it2 != shape2.rend()) ? *it2++ : 1;

            if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                throw std::invalid_argument("Imcompatible shape for broadcasting!");

            result.push_back(std::max(dim1, dim2));
        }

        std::reverse(result.begin(), result.end());
        return result;
    }

    // Add one to the shape while broadcasting
    std::vector<size_t> _extend_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
        std::vector<size_t> result;
        size_t dif;
        if (shape1.size() > shape2.size()) {
            result = shape2;
            dif = shape1.size() - shape2.size();
        }
        else {
            result = shape1;
            dif = shape2.size() - shape1.size();
        }

        for (int i = 0; i < dif; i++)
            result.insert(result.begin(), 1);

        return result;
    }

}