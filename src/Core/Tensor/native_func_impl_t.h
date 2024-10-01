//
// Created by zzy on 2024/8/8.
//

#ifndef NEURALNETWORK_BASE_NATIVE_FUNC_IMPL_T_H
#define NEURALNETWORK_BASE_NATIVE_FUNC_IMPL_T_H

#include "tensor.h"

namespace NN {
    template <typename T>
    Tensor<T> pow(const Tensor<T>& a, const Tensor<T>& b) {
        if (b.get_shape().size() != 1 && b.get_shape()[0] != 1)
            throw std::runtime_error("Trying to power the tensor with the shape of the exponential number more than 1");

        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        T power = *(b.begin());
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = std::pow(*ptr_a, power);
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> pow(const Tensor<T>& a, const T& b) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = std::pow(*ptr_a, b);
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> log(const Tensor<T>& a) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = std::log(*ptr_a);
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> exp(const Tensor<T>& a) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = std::exp(*ptr_a);
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> sin(const Tensor<T>& a) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = std::sin(*ptr_a);
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> cos(const Tensor<T>& a) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = std::cos(*ptr_a);
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> tan(const Tensor<T>& a) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = std::tan(*ptr_a);
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> sec(const Tensor<T>& a) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = 1 / std::cos(*ptr_a);
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> sum(const Tensor<T>& a) {
        Tensor<T> res((TensorShape({1})));
        T* ptr = res.begin();
        for (T* ptr_a = a.begin(); ptr_a != a.end(); ptr_a++) {
            *ptr += *ptr_a;
        }

        return res;
    }
}



#endif //NEURALNETWORK_BASE_NATIVE_FUNC_IMPL_T_H
