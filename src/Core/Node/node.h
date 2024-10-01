//
// Created by zzy on 2024/8/6.
//

#ifndef NEURALNETWORK_BASE_NODE_H
#define NEURALNETWORK_BASE_NODE_H

#include "base_node.h"
#include "../Tensor/native_func_impl_t.h"
#include "../Tensor/activation_func_impl_t.h"
#include "../Tensor/loss_func_impl_t.h"
#include <string>

namespace NN {
    template <typename T>
    class AddNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 2)
                throw std::invalid_argument("Invalid number of input for add operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(inputs[0]->_data + inputs[1]->_data, this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                Tensor<T> gy = *(output_ptr->_grad);
                if (this->_inputs[0]->requires_grad()) {
                    if (this->_inputs[0]->_grad->size() != gy.size())
                        gy = gy.sum_to(this->_inputs[0]->_grad->get_shape());
                    *(this->_inputs[0]->_grad) += gy;
                }

                if (this->_inputs[1]->requires_grad()) {
                    if (this->_inputs[1]->_grad->size() != gy.size())
                        gy = gy.sum_to(this->_inputs[1]->_grad->get_shape());
                    *(this->_inputs[1]->_grad) += gy;
                }
            }
        }
    };

    template <typename T>
    class SubNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 2)
                throw std::invalid_argument("Invalid number of input for sub operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(inputs[0]->_data - inputs[1]->_data, this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                Tensor<T> gy = *(output_ptr->_grad);
                if (this->_inputs[0]->requires_grad()) {
                    if (this->_inputs[0]->_grad->size() != gy.size())
                        gy = gy.sum_to(this->_inputs[0]->_grad->get_shape());
                    *(this->_inputs[0]->_grad) += gy;
                }

                if (this->_inputs[1]->requires_grad()) {
                    if (this->_inputs[1]->_grad->size() != gy.size())
                        gy = gy.sum_to(this->_inputs[1]->_grad->get_shape());
                    *(this->_inputs[1]->_grad) -= gy;
                }
            }
        }
    };

    template <typename T>
    class MulNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 2)
                throw std::invalid_argument("Invalid number of input for mul operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(inputs[0]->_data * inputs[1]->_data, this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    auto gx = *(output_ptr->_grad) * this->_inputs[1]->_data;
                    if (this->_inputs[0]->_grad->size() != gx.size())
                        gx = gx.sum_to(this->_inputs[0]->_grad->get_shape());
                    *(this->_inputs[0]->_grad) += gx;
                }

                if (this->_inputs[1]->requires_grad()) {
                    auto gx = *(output_ptr->_grad) * this->_inputs[0]->_data;
                    if (this->_inputs[1]->_grad->size() != gx.size())
                        gx = gx.sum_to(this->_inputs[1]->_grad->get_shape());
                    *(this->_inputs[1]->_grad) += gx;
                }
            }
        }
    };

    template <typename T>
    class DivNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 2)
                throw std::invalid_argument("Invalid number of input for div operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(inputs[0]->_data / inputs[1]->_data, this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    auto gx = (output_ptr->_grad) / this->_inputs[1]->_data;
                    if (gx.size() != this->_inputs[0]->_grad->size())
                        gx = gx.sum_to(this->_inputs[0]->_grad->get_shape());
                    *(this->_inputs[0]->_grad) += gx;
                }

                if (this->_inputs[1]->requires_grad()) {
                    auto gx = *(output_ptr->_grad) * this->_inputs[0]->_data;
                    if (gx.size() != this->_inputs[1]->_grad->size())
                        gx = gx.sum_to(this->_inputs[1]->_grad->get_shape());
                    *(this->_inputs[1]->_grad) -= gx / (this->_inputs[1]->_data * this->_inputs[1]->data);
                }
            }
        }
    };

    template <typename T>
    class PowNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 2)
                throw std::invalid_argument("Invalid number of input for pow operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(pow(inputs[0]->_data, inputs[1]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += *(output_ptr->_grad) * this->_inputs[1]->_data * pow(this->_inputs[0]->_data, this->_inputs[1]->_data - Tensor<T>::ones(this->_inputs[1]->_data.get_shape()));
                }

                if (this->_inputs[1]->requires_grad()) {
                    *(this->_inputs[1]->_grad) -= *(output_ptr->_grad) * output_ptr->_data * log(this->_inputs[0]->_data);
                }
            }
        }
    };

    template <typename T>
    class ExpNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for exp operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(exp(inputs[0]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += *(output_ptr->_grad) * output_ptr->_data;
                }
            }
        }
    };

    template <typename T>
    class LogNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for log operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(log(inputs[0]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += *(output_ptr->_grad) / this->_inputs[0]->_data;
                }
            }
        }
    };

    template <typename T>
    class SinNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for sin operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(sin(inputs[0]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += *(output_ptr->_grad) * cos(this->_inputs[0]->_data);
                }
            }
        }
    };

    template <typename T>
    class CosNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for cos operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(cos(inputs[0]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += *(output_ptr->_grad) * -1 * sin(this->_inputs[0]->_data);
                }
            }
        }
    };

    template <typename T>
    class TanNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for tan operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(tan(inputs[0]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += *(output_ptr->_grad) * sec(this->_inputs[0]->_data) * sec(this->_inputs[0]->_data);
                }
            }

        }
    };

    template <typename T>
    class SumNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for sum operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(sum(inputs[0]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += broadcast_data(*(output_ptr->_grad), *(this->_inputs[0]->_grad), nullptr);
                }
            }
        }
    };

    template <typename T>
    class SigmoidNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for sigmoid operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(sigmoid(inputs[0]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += *(output_ptr->_grad) * sigmoid(this->_inputs[0]->_data) * (1 - sigmoid(this->_inputs[0]->_data));
                }
            }
        }
    };

    template <typename T>
    class TanhNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for tanh operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(tanh(inputs[0]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += *(output_ptr->_grad) * (1 - output_ptr->get_data() * output_ptr->get_data());
                }
            }
        }
    };

    template <typename T>
    class ReLUNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for tanh operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(relu(inputs[0]->_data, _mask), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    *(this->_inputs[0]->_grad) += relu_grad(this->_inputs[0]->_data) * *(output_ptr->_grad);
                }
            }
        }

    private:
        bool _mask = false;
    };

    template <typename T>
    class SoftmaxNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for softmax operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            // std::cout << inputs[0]->_data << std::endl;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(softmax(inputs[0]->_data, 1), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    // input -> (60000, 10)
                    // 10 -> 10 (p)
                    // grad -> (60000, 10)

                    *(this->_inputs[0]->_grad) += softmax_grad_func(output_ptr->_data, *(output_ptr->_grad));
                    // std::cout << *(this->_inputs[0]->_grad) << std::endl;
                }
            }
        }

    };

    template <typename T>
    class LogSoftmaxNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 1)
                throw std::invalid_argument("Invalid number of input for log_softmax operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            // std::cout << "Softmax input: " << inputs[0]->_data << std::endl;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(log_softmax(inputs[0]->_data, 1), this, true);
            this->_output = output_ptr;
            // std::cout << output_ptr->_data << std::endl;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    // input -> (60000, 10)
                    // 10 -> 10 (p)
                    // grad -> (60000, 10)

                    *(this->_inputs[0]->_grad) += log_softmax_grad(this->_inputs[0]->_data, *(output_ptr->_grad), 1);
                    std::cout << "Softmax_grad: " << *(this->_inputs[0]->_grad) << std::endl;
                }
            }
        }

    };

    template <typename T>
    class MSELossNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 2)
                throw std::invalid_argument("Invalid number of input for MSE operation! Current input variable count is " + std::to_string(inputs.size()));

            // _inputs[0]->label, _inputs[1]->prediction
            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(((inputs[0]->_data - inputs[1]->_data) * (inputs[0]->_data - inputs[1]->_data)).mean(), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[1]->requires_grad()) {
                    auto gy = *(output_ptr->_grad) * 2 * (this->_inputs[1]->_data - this->_inputs[0]->_data) / this->_inputs[0]->_data.size();
                    *(this->_inputs[1]->_grad) += gy;
                }
            }
        }
    };

    template <typename T>
    class NLLLossNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 2)
                throw std::invalid_argument("Invalid number of input for MSE operation! Current input variable count is " + std::to_string(inputs.size()));

            // _inputs[0]->label_idx, _inputs[1]->prediction
            this->_inputs = inputs;
            // Retrieve the target index
            _indices = inputs[0]->_data.to_std_vector();
//            for (int i = 0; i < 100; i++)
//                std::cout << (int)_indices[i] << ", ";
//            std::cout << "\n";
            //std::cout << "Loss Input: " << inputs[1]->_data << "\n";
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(nll_loss_func<T>(_indices, inputs[1]->_data), this, true);
            this->_output = output_ptr;
            // std::cout << "Loss output: " << output_ptr->_data << "\n";
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[1]->requires_grad()) {
                    *(this->_inputs[1]->_grad) += nll_loss_func_grad<T>(_indices, this->_inputs[1]->_data.get_shape()) * *(output_ptr->_grad);
                    // std::cout << "NLL_Loss_Grad: " <<  *(this->_inputs[1]->_grad) << std::endl;
                }
            }
        }

    private:
        std::vector<T> _indices;
    };

    template <typename T>
    class MatMulNode : public com_node<T> {
    public:
        std::shared_ptr<data_node<T>> forward(std::vector<std::shared_ptr<data_node<T>>> inputs) override {
            if (inputs.size() != 2)
                throw std::invalid_argument("Invalid number of input for mat_mul operation! Current input variable count is " + std::to_string(inputs.size()));

            this->_inputs = inputs;
            std::shared_ptr<data_node<T>> output_ptr = std::make_shared<data_node<T>>(inputs[0]->_data.matmul(inputs[1]->_data), this, true);
            this->_output = output_ptr;
            return output_ptr;
        }

        void backward() override {
            if (auto output_ptr = this->_output.lock()) {
                if (this->_inputs[0]->requires_grad()) {
                    auto gx = output_ptr->_grad->matmul(this->_inputs[1]->_data.transpose());
//                    if (this->_inputs[0]->_grad->size() != gx.size())
//                        gx = gx.sum_to(this->_inputs[0]->_grad->get_shape());
                    *(this->_inputs[0]->_grad) += gx;
                }

                if (this->_inputs[1]->requires_grad()) {
                    auto gx = this->_inputs[0]->_data.transpose().matmul(*(output_ptr->_grad));
//                    if (this->_inputs[1]->_grad->size() != gx.size())
//                        gx = gx.sum_to(this->_inputs[1]->_grad->get_shape());
                    *(this->_inputs[1]->_grad) += gx;
                    // std::cout << *(this->_inputs[1]->_grad) << std::endl;
                }
            }
        }
    };
}

#endif //NEURALNETWORK_BASE_NODE_H
