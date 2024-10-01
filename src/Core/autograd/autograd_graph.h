//
// Created by zzy on 2024/8/6.
//

#ifndef NEURALNETWORK_BASE_AUTOGRAD_GRAPH_H
#define NEURALNETWORK_BASE_AUTOGRAD_GRAPH_H

#include "../Node/node.h"
#include <queue>

namespace NN {
    template <typename T>
    class autograd_graph {
    public:
        explicit autograd_graph(const std::shared_ptr<data_node<T>> &output) : output(output) {}

        ~autograd_graph() {
            grad_list.clear();
        }

        void backward() {
            if (output == nullptr) {
                std::cerr << "output variable of the autograd_graph is null, please assign it before executing backward propagation." << std::endl;
                return;
            }

            // 设置输出节点的grad为1，未来允许多次
            output->_grad->ones();
            std::queue<std::shared_ptr<data_node<T>>> q_front;
            q_front.push(output);
            while (!q_front.empty()) {
                std::shared_ptr<data_node<T>> cur_node = q_front.front();
                q_front.pop();

                // push grad into the grad_list
                grad_list.push_back(cur_node->_grad);

                if (cur_node->_grad_func == nullptr) // if leaf node
                    continue;

                cur_node->_grad_func->backward();
                for (auto n : cur_node->_grad_func->get_input()) {
                    q_front.push(n);
                }
            }
        }

        void zero_grad() {
            for (auto ptr : grad_list) {
                if (auto grad = ptr.lock()) {
                    grad->zeros();
                }
            }
        }

    private:
        std::shared_ptr<data_node<T>> output;
        std::vector<std::weak_ptr<Tensor<T>>> grad_list;
    };
}





#endif //NEURALNETWORK_BASE_AUTOGRAD_GRAPH_H
