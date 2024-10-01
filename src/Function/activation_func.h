//
// Created by zzy on 2024/8/9.
//

#ifndef NEURALNETWORK_BASE_ACTIVATION_FUNC_H
#define NEURALNETWORK_BASE_ACTIVATION_FUNC_H

#include <initializer_list>
#include <vector>
#include <stdexcept>
#include "../Variable/Variable.h"

namespace NN {
    template <typename T>
    Variable<T> sigmoid(const Variable<T>& x) {
        SigmoidNode<T>* sigNode = new SigmoidNode<T>();
        return Variable<T>(sigNode->forward({x.get_node()}));
    }

    template <typename T>
    Variable<T> tanh(const Variable<T>& x) {
        TanhNode<T>* tanHNode = new TanhNode<T>();
        return Variable<T>(tanHNode->forward({x.get_node()}));
    }

    template <typename T>
    Variable<T> relu(const Variable<T>& x) {
        ReLUNode<T>* reluNode = new ReLUNode<T>();
        return Variable<T>(reluNode->forward({x.get_node()}));
    }

    template <typename T>
    Variable<T> softmax(const Variable<T>& x) {
        SoftmaxNode<T>* softmaxNode = new SoftmaxNode<T>();
        return Variable<T>(softmaxNode->forward({x.get_node()}));
    }

    template <typename T>
    Variable<T> logSoftmax(const Variable<T>& x) {
        LogSoftmaxNode<T>* softmaxNode = new LogSoftmaxNode<T>();
        return Variable<T>(softmaxNode->forward({x.get_node()}));
    }
}

#endif //NEURALNETWORK_BASE_ACTIVATION_FUNC_H
