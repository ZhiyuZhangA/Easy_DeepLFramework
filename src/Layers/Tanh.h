//
// Created by zzy on 2024/8/17.
//

#ifndef NEURALNETWORK_BASE_TANH_H
#define NEURALNETWORK_BASE_TANH_H

#include "LayerBase.h"
#include "../Function/Function.h"

namespace NN {
    template <typename T>
    class Tanh : public LayerBase<T> {
    public:
        Variable<T> forward(const Variable<T>& x) const override {
            return tanh(x);
        }
    };
}

#endif //NEURALNETWORK_BASE_TANH_H
