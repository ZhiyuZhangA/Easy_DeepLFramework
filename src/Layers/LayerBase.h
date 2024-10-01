//
// Created by zzy on 2024/8/13.
//

#ifndef NEURALNETWORK_BASE_LAYERBASE_H
#define NEURALNETWORK_BASE_LAYERBASE_H

#include "../Variable/Variable.h"

namespace NN {

    template <typename T>
    class LayerBase {
    public:
        virtual Variable<T> forward(const Variable<T>& x) const = 0;

        std::vector<std::shared_ptr<Variable<T>>> get_parameters() { return _params; }

    protected:
        std::vector<std::shared_ptr<Variable<T>>> _params;

    };
}



#endif //NEURALNETWORK_BASE_LAYERBASE_H
