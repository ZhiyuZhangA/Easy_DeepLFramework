//
// Created by zzy on 2024/8/10.
//

#ifndef NEURALNETWORK_BASE_LOSS_FUNC_H
#define NEURALNETWORK_BASE_LOSS_FUNC_H

#include "../Variable/Variable.h"
#include "common_func.h"

namespace NN {
    template <typename T>
    Variable<T> mse_loss(const Variable<T>& label, const Variable<T>& prediction) {
        MSELossNode<T>* mseNode = new MSELossNode<T>();
        return Variable<T>(mseNode->forward({label.get_node(), prediction.get_node()}));
    }

    template <typename T>
    Variable<T> nll_loss(const Variable<T>& label_idx, const Variable<T>& prediction) {
        NLLLossNode<T>* nllloseNode = new NLLLossNode<T>();
        return Variable<T>(nllloseNode->forward({label_idx.get_node(), prediction.get_node()}));
    }
}


#endif //NEURALNETWORK_BASE_LOSS_FUNC_H
