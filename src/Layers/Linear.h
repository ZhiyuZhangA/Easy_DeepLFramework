//
// Created by zzy on 2024/8/13.
//

#ifndef NEURALNETWORK_BASE_LINEAR_H
#define NEURALNETWORK_BASE_LINEAR_H

#include "LayerBase.h"

namespace NN {
    template <typename T>
    class Linear : public LayerBase<T> {
    public:
        Linear(size_t in_features, size_t out_features, bool bias=true) : _in_features(in_features), _out_features(out_features), _bias(bias) {
            Variable<T> var1((Tensor<T>(TensorShape({in_features, out_features}))), true);
            float k = 1.0 / in_features;
            float l_b = -std::sqrt(k);
            float u_b = std::sqrt(k);
            var1.uniform_real(l_b, u_b);
            // var1.xavier_uniform(in_features, out_features);
            std::shared_ptr<Variable<T>> weight = std::make_shared<Variable<T>>(var1);
            this->_params.push_back(weight);
            if (bias) {
                Variable<T> var2((Tensor<T>(TensorShape({out_features}))), true);
                var2.uniform_real(l_b, u_b);
                std::shared_ptr<Variable<T>> b = std::make_shared<Variable<T>>(var2);
                this->_params.push_back(b);
            }
        }

        Variable<T> forward(const Variable<T>& x) const override {
            auto xw = x.matmul(*(this->_params[0]));
            if (_bias)
                xw = xw + *(this->_params[1]);
            // std::cout << *(this->_params[0]->get_grad()) << "\n";
            return xw;
        }

        Variable<T> get_weight() {
            return *(this->_params[0]);
        }

        void set_weight(std::shared_ptr<Variable<T>> var) {
            this->_params[0] = var;
        }

        Variable<T> get_bias() {
            if (!_bias)
                throw std::runtime_error("Error occurred at " + std::string(__FILE__) + " Line " + std::to_string(__LINE__) +  " Linear LayerBase doesn't have a bias term!");
            return *(this->_params[1]);
        }

    private:

        size_t _in_features;
        size_t _out_features;
        bool _bias;
    };
}




#endif //NEURALNETWORK_BASE_LINEAR_H
