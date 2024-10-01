//
// Created by zzy on 2024/8/17.
//

#ifndef NEURALNETWORK_BASE_MODEL_H
#define NEURALNETWORK_BASE_MODEL_H

#include <vector>
#include "../Layers/LayerBase.h"

namespace NN {

    template <typename T>
    class Model {
    public:
        Model(const std::vector<std::shared_ptr<LayerBase<T>>>& il) {
            _layers = il;
            for (auto l : _layers) {
                auto vec = l->get_parameters();
                _params.insert(_params.end(), vec.begin(), vec.end());
            }
        }

        Variable<T> operator()(const Variable<T>& input) {
            Variable<T> res = input;
            for (auto l : _layers) {
                res = l->forward(res);
            }

            return res;
        }

        std::vector<std::shared_ptr<Variable<T>>> get_parameters() const { return _params; }

    private:
        std::vector<std::shared_ptr<LayerBase<T>>> _layers;
        std::vector<std::shared_ptr<Variable<T>>> _params;
    };

    template <typename T>
    Model<T> make_model(const std::vector<std::shared_ptr<LayerBase<T>>>& il) {
        return Model(il);
    }
}

#endif //NEURALNETWORK_BASE_MODEL_H
