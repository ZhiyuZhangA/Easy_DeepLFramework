//
// Created by zzy on 2024/8/9.
//

#ifndef NEURALNETWORK_BASE_COMMON_FUNC_H
#define NEURALNETWORK_BASE_COMMON_FUNC_H

#include <initializer_list>
#include <vector>
#include <stdexcept>
#include "../Variable/Variable.h"

namespace NN {
    template <typename T>
    Variable<T> pow(const Variable<T>& a, const Variable<T>& b) {
        PowNode<T>* powNode = new PowNode<T>();
        return Variable<T>(powNode->forward({a.get_node(), b.get_node()}));
    }

    template <typename T>
    Variable<T> log(const Variable<T>& a) {
        LogNode<T>* logNode = new LogNode<T>();
        return Variable<T>(logNode->forward({a.get_node()}));
    }

    template <typename T>
    Variable<T> exp(const Variable<T>& a) {
        ExpNode<T>* expNode = new ExpNode<T>();
        return Variable<T>(expNode->forward({a.get_node()}));
    }

    template <typename T>
    Variable<T> sin(const Variable<T>& a) {
        SinNode<T>* sinNode = new SinNode<T>();
        return Variable<T>(sinNode->forward({a.get_node()}));
    }

    template <typename T>
    Variable<T> cos(const Variable<T>& a) {
        CosNode<T>* cosNode = new CosNode<T>();
        return Variable<T>(cosNode->forward({a.get_node()}));
    }

    template <typename T>
    Variable<T> tan(const Variable<T>& a) {
        TanNode<T>* tanNode = new TanNode<T>();
        return Variable<T>(tanNode->forward({a.get_node()}));
    }

    template <typename T>
    Variable<T> sum(const Variable<T>& a) {
        SumNode<T>* sumNode = new SumNode<T>();
        return Variable<T>(sumNode->forward({a.get_node()}));
    }

}

#endif //NEURALNETWORK_BASE_COMMON_FUNC_H
