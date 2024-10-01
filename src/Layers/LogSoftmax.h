//
// Created by zzy on 2024/8/21.
//

#ifndef NEURALNETWORK_BASE_LOGSOFTMAX_H
#define NEURALNETWORK_BASE_LOGSOFTMAX_H

#include "LayerBase.h"
#include "../Function/Function.h"

namespace NN {
    template <typename T>
    class LogSoftmax : public LayerBase<T> {
    public:
        Variable<T> forward(const Variable<T>& x) const override {
            return logSoftmax(x);
        }
    };
}

#endif //NEURALNETWORK_BASE_LOGSOFTMAX_H
