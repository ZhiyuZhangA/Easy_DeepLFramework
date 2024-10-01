//
// Created by zzy on 2024/8/6.
//

#ifndef NEURALNETWORK_BASE_BASE_NODE_H
#define NEURALNETWORK_BASE_BASE_NODE_H

#include "../Tensor/tensor.h"
#include <vector>

namespace NN {

    template <typename T>
    class com_node;

    template <typename T>
    class data_node {
    public:
        data_node() = default;

        data_node(Tensor<T> data, com_node<T>* grad_func, bool requires_grad) : _data(data), _grad_func(grad_func), _requires_grad(requires_grad) {
            if (requires_grad)
                _grad = std::make_shared<Tensor<T>>(_data.get_shape());
            else
                _grad = nullptr;
        }

        data_node(Tensor<T> data, bool requires_grad) : data_node(data, nullptr, requires_grad) { }

        // Tensor类里面需要一个non-const lvalue，而我这里返回的是rvalue，导致在做加法的时候无法进行.
        inline Tensor<T>& get_data() { return _data; }

        bool requires_grad() const { return _requires_grad; }

        void set_requires_grad(bool req=true) {
            // Whether the data node is the leaf node
            if (_grad_func != nullptr) {
                throw std::runtime_error("Only allowed to change the property requires_grad for leaf node!");
            }

            _requires_grad = req;
            if (!_grad)
                _grad = std::make_shared<Tensor<T>>(_data.get_shape());
        }

        Tensor<T> _data;
        std::shared_ptr<Tensor<T>> _grad;
        std::shared_ptr<com_node<T>> _grad_func; // pre-node

    private:
        bool _requires_grad;
    };

    template <typename T>
    class com_node {
    public:
        com_node() = default;
        virtual ~com_node() { };
        virtual void backward() = 0;
        virtual std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) = 0;

        void set_output(data_node<T>* output) { this->_output = output; }

        std::vector<std::shared_ptr<data_node<T>>> get_input() { return _inputs; }

    protected:
        std::vector<std::shared_ptr<data_node<T>>> _inputs;
        std::weak_ptr<data_node<T>> _output;
    };

}

#endif //NEURALNETWORK_BASE_BASE_NODE_H
