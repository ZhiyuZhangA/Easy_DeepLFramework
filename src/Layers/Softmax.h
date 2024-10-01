//
// Created by zzy on 2024/8/20.
//

#ifndef NEURALNETWORK_BASE_SOFTMAX_H
#define NEURALNETWORK_BASE_SOFTMAX_H

#include "LayerBase.h"
#include "../Function/Function.h"

namespace NN {
    template <typename T>
    class Softmax : public LayerBase<T> {
    public:
        Variable<T> forward(const Variable<T>& x) const override {
            return softmax(x);
        }
    };
}

#endif //NEURALNETWORK_BASE_SOFTMAX_H
