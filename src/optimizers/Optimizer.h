//
// Created by zzy on 2024/8/10.
//

#ifndef NEURALNETWORK_BASE_OPTIMIZER_H

#include <vector>
#include "../Variable/Variable.h"

namespace NN {
    template <typename T>
    class Optimizer {
    public:
        Optimizer(const std::vector<std::shared_ptr<Variable<T>>>& params, const double& lr) : _lr(lr) {
            convert_params(params);
        }

        virtual void step() { };

        // Clear the gradients
        void zero_grads() {
            for (auto p : _params) {
                if (auto var = p.lock()) {
                    var->zero_grad();
                }
            }
        }

        void set_lr(const double& lr) {
            this->_lr = lr;
        }

    protected:

        // Learning Rate
        double _lr;

        // Params List
        std::vector<std::weak_ptr<Variable<T>>> _params;

        void convert_params(const std::vector<std::shared_ptr<Variable<T>>>& params) {
            for (auto ptr : params) {
                _params.push_back(ptr);
            }
        }

    };

    template <typename T>
    class SGD : public Optimizer<T> {
    public:
        SGD(const std::vector<std::shared_ptr<Variable<T>>>& params, const double& lr=0.01) : Optimizer<T>(params, lr) { }

        void step() override {
            for (auto ptr : this->_params) {
                if (auto var = ptr.lock()) {
                    if (var->requires_grad()) {
                        auto tensor = var->get_data();
                        tensor = tensor - (*(var->get_grad()) * (T)this->_lr);
                        var->set_data(tensor);
                    }
                }
            }
        }
    };
}



#endif //NEURALNETWORK_BASE_OPTIMIZER_H
