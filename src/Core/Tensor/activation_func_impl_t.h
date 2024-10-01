//
// Created by zzy on 2024/8/9.
//

#ifndef NEURALNETWORK_BASE_ACTIVATION_FUNC_IMPL_T_H
#define NEURALNETWORK_BASE_ACTIVATION_FUNC_IMPL_T_H

#include "tensor.h"
#include "numeric"

namespace NN {
    template <typename T>
    Tensor<T> sigmoid(const Tensor<T>& a) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = 1 / (1 + std::exp(*ptr_a * -1));
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> tanh(const Tensor<T>& a) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            // *ptr = (std::exp(*ptr_a) - std::exp(*ptr_a * (-1))) / (std::exp(*ptr_a) + std::exp(*ptr_a * (-1)));
            *ptr = std::tanh(*ptr_a);
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> relu(const Tensor<T>& a, bool& mask) {
        Tensor<T> res(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            *ptr = *ptr_a > 0 ? *ptr_a : 0;
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> relu_grad(const Tensor<T>& a) {
        Tensor<T> grad(a.get_shape());
        T* ptr_a = a.begin();
        for (T* ptr = grad.begin(); ptr != grad.end(); ptr++) {
            *ptr = *ptr_a > 0 ? *ptr_a : 0;
            if (*ptr != *ptr_a)
                *ptr = 1;
            ptr_a++;
        }

        return grad;
    }

    template <typename T>
    Tensor<T> softmax(const Tensor<T>& a, size_t axis) {
        Tensor<T> res(a.get_shape());
        auto target_dim = res.get_shape()[axis];
        T* ptr_a = a.begin();
        int idx = 0;
        T sum = 0;
        for (T* ptr = res.begin(); ptr != res.end(); ptr++) {
            if (idx % target_dim == 0 && idx < res.size() - 1) {
                sum = 0; // clear the sum
                for (T* tmp_ptr = ptr_a; tmp_ptr != ptr_a + target_dim; tmp_ptr++) {
                    sum += std::exp(*tmp_ptr);
                }
            }
            *ptr = std::exp(*ptr_a) / sum;
            idx++;
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> softmax_grad_func(const Tensor<T>& y, const Tensor<T>& _grad_in) {
        Tensor<T> grad(y.get_shape());

//        // Create the Jacobin Matrix
//        int n = grad.get_shape()[grad.get_dim() - 1];
//        std::vector<T> j_matrix;
//        T* s_ptr = y.begin();
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < n; j++) {
//                if (i == j)
//                    j_matrix[i * n + j] = *(s_ptr + )
//            }
//        }

        auto batch_size = y.get_shape()[0];
        auto n = y.get_shape()[1];
        T* ptr_y = y.begin();          // pointer to the softmax result
        T* ptr_g_i = _grad_in.begin(); // pointer to the input gradient
        T* ptr_g = grad.begin();       // pointer to the grad computed
        for (auto i = 0; i < batch_size; i++) {
            T sum = 0;
            for (auto j = 0; j < n; j++) {
                sum += *(ptr_y + j) * *(ptr_g_i + j);
            }

            for (auto j = 0; j < n; j++) {
                *ptr_g = *(ptr_y + j) * (*ptr_g_i - sum);
                ptr_g++;
            }
            ptr_y += n;
            ptr_g_i += n;
        }

        return grad;
    }

    template <typename T>
    Tensor<T> log_softmax(const Tensor<T>& a, int axis) {
        Tensor<T> res(a.get_shape());
        auto target_dim = res.get_shape()[axis];
        T *ptr_a = a.begin();
        int idx = 0;
        T sum = 0;
        for (T *ptr = res.begin(); ptr != res.end(); ptr++) {
            if (idx % target_dim == 0 && idx < res.size() - 1) {
                sum = 0; // clear the sum
                for (T *tmp_ptr = ptr_a; tmp_ptr != ptr_a + target_dim; tmp_ptr++) {
                    sum += std::exp(*tmp_ptr);
                }
                sum = std::log(sum);
            }

            *ptr = *ptr_a - sum;
            idx++;
            ptr_a++;
        }

        return res;
    }

    template <typename T>
    Tensor<T> log_softmax_grad(const Tensor<T>& a, const Tensor<T>& grad_in, int axis) {
        Tensor<T> softmax_val = softmax(a, axis);
        Tensor<T> grad(a.get_shape());
        int batch = a.get_shape()[0];
        int n = a.get_shape()[axis];
        T* ptr_g = grad.begin();
        T* ptr_s = softmax_val.begin();
        T* ptr_g_in = grad_in.begin();

        for (int i = 0; i < batch; i++) {
            T sum_g_in = std::accumulate(ptr_g_in, ptr_g_in + n, 0.0);
            for (int j = 0; j < n; j++) {
                *ptr_g = *ptr_g_in - *ptr_s * sum_g_in;
                ptr_g++;
                ptr_g_in++;
                ptr_s++;
            }
        }

        return grad;
    }
}


#endif //NEURALNETWORK_BASE_ACTIVATION_FUNC_IMPL_T_H
