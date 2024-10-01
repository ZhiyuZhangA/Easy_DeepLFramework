//
// Created by zzy on 2024/8/17.
//

#ifndef NEURALNETWORK_BASE_RELU_H
#define NEURALNETWORK_BASE_RELU_H

#include "LayerBase.h"
#include "../Function/Function.h"

namespace NN {
    template <typename T>
    class ReLU : public LayerBase<T> {
    public:
        Variable<T> forward(const Variable<T>& x) const override {
            return relu(x);
        }
    };
}

#endif //NEURALNETWORK_BASE_TANH_H
