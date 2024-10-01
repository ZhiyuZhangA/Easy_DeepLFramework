//
// Created by zzy on 2024/8/20.
//

#ifndef NEURALNETWORK_BASE_LOSS_FUNC_IMPL_T_H
#define NEURALNETWORK_BASE_LOSS_FUNC_IMPL_T_H

#include "tensor.h"

namespace NN {
    template <typename T>
    Tensor<T> nll_loss_func(const std::vector<T>& indices, const Tensor<T>& raw_data) {
        auto n = indices.size(); // batch size
        Tensor<T> res((TensorShape({1})));
        T* ptr = res.begin();
        T* ptr_r = raw_data.begin();
        auto c = raw_data.get_shape()[1];
        T l = -1.0 / n;
        for (auto i = 0; i < n; i++) {
            // *ptr += -std::log(*(ptr_r + (int)indices[i]));
            *ptr += *(ptr_r + (int)indices[i]) * l;
            ptr_r += c;
        }

        return res;
    }

    template <typename T>
    Tensor<T> nll_loss_func_grad(const std::vector<T>& indices, const std::vector<size_t>& shape) {
        Tensor<T> grad(shape);
        int n = shape[0];
        int col = shape[1];
        T l = -1.0 / (float)n;
        T* ptr = grad.begin();
        for (int i = 0; i < n; i++) {
            *(ptr + (int)indices[i]) = l;
            ptr += col;
        }

        // std::cout << grad << std::endl;
        return grad;
    }
}


#endif //NEURALNETWORK_BASE_LOSS_FUNC_IMPL_T_H
