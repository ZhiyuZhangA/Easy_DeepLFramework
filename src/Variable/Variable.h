#ifndef _VARIABLE_
#define _VARIABLE_

#include "../Core/autograd/autograd_graph.h"
#include <vector>
#include <random>

namespace NN {

    template <typename T>
    class Variable {
    public:
        Variable(const Tensor<T>& tensor, bool requires_grad=false) {
            _data_node = std::make_shared<data_node<T>>(tensor, requires_grad);
        }

        Variable(std::shared_ptr<data_node<T>> data_node) : _data_node(data_node) {
        }

        ~Variable() {

        }

        void backward() {
            if (_data_node->get_data().get_dim() > 1) {
                throw std::runtime_error("grad can be implicitly created only for scalar outputs");
            }

            _graph = std::make_shared<autograd_graph<T>>(_data_node);
            _graph->backward();
        }

        void zero_grad() {
            if (requires_grad()) {
                _data_node->_grad->zeros();
            }
            else {
                std::cerr << "Trying to clear the gradient for variable that doesn't require gradient!" << std::endl;
            }
        }


        Variable<T> operator+(const Variable<T>& other) {
            AddNode<T>* addNode = new AddNode<T>();
            return Variable<T>(addNode->forward({(this->_data_node), (other._data_node)}));
        }

        Variable<T> operator-(const Variable<T>& other) const {
            SubNode<T>* subNode = new SubNode<T>();
            return Variable<T>(subNode->forward({(this->_data_node), (other._data_node)}));
        }

        Variable<T> operator*(const Variable<T>& other) const {
            MulNode<T>* mulNode = new MulNode<T>();
            return Variable<T>(mulNode->forward({(this->_data_node), (other._data_node)}));
        }

        Variable<T> operator/(const Variable<T>& other) {
            DivNode<T>* divNode = new DivNode<T>();
            return Variable<T>(divNode->forward({(this->_data_node), (other._data_node)}));
        }

        Variable<T> matmul(const Variable<T>& other) const {
            MatMulNode<T>* mat_mulNode = new MatMulNode<T>();
            return Variable<T>(mat_mulNode->forward({this->_data_node, other._data_node}));
        }

        Variable<T> sum() const {
            SumNode<T>* sum_node = new SumNode<T>();
            return Variable<T>(sum_node->forward({this->_data_node}));
        }

        inline bool requires_grad() { return _data_node->requires_grad(); }
        inline Tensor<T> get_data() { return _data_node->_data; }
        inline std::shared_ptr<Tensor<T>> get_grad() { return (_data_node->_grad); }
        std::shared_ptr<data_node<T>> get_node() const { return _data_node; }

        void set_data(const Tensor<T>& data) {
            _data_node->_data = data;
        }

        void xavier_uniform(int n_0, int n_1) {
            std::vector<T> filled;
            std::mt19937 gen(std::random_device{}());
            std::normal_distribution<T> dist(0, sqrt( 2.0 / (float)(n_0 + n_1)));
            for (int i = 0; i < _data_node->_data.size(); i++) {
                filled.push_back(dist(gen));
            }

            _data_node->_data.fill(filled);
        }

        void uniform_real(float lower_bound, float upper_bound) {
            std::random_device rd;
            std::default_random_engine generator(rd());
            std::uniform_real_distribution<T> distribution(lower_bound, upper_bound);
            std::vector<T> filled;
            for (int i = 0; i < _data_node->_data.size(); i++) {
                filled.push_back(distribution(generator));
            }

            _data_node->_data.fill(filled);
        }

    private:
        std::shared_ptr<data_node<T>> _data_node;
        std::shared_ptr<autograd_graph<T>> _graph;
    };

}



#endif