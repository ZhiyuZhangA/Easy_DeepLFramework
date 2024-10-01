#include <iostream>
#include "Variable/Variable.h"
#include "Function/Function.h"
#include "optimizers/Optimizer.h"
#include "Core/autograd/autograd_graph.h"
#include "Layers/Layer.h"
#include "Model/Model.h"
#include "Dataset/MNIST.h"

using namespace std;
using namespace NN;

//void tensor_matmul_test() {
//    size_t N = 2048;
//
//    std::vector<size_t> shape = {N, N};
//    TensorShape t_shape(shape);
//    Tensor<float32> tensor1(t_shape);
//    tensor1.fill(arange_vec<float32>(N * N + 1, 1));
//    Tensor<float32> tensor2(t_shape);
//    tensor2.fill(arange_vec<float32>(N * N + 1, 1));`
//
//    clock_t start = clock();
//    Tensor<float32> result = tensor1.matmul(tensor2);
//    clock_t end = clock();
//
//    double dur = (double)(end - start) / CLOCKS_PER_SEC;
//
//    cout <<  dur << " GFLOP/S" << endl;
//}

void test(const Variable<float32>& input, const Variable<float32>& label, Model<float32>& model) {
    cout << input.get_node()->get_data().get_tensorShape() << endl;
    Variable<float32> prediction = model(input);
    auto data = prediction.get_data();
    float correct_n;
    auto label_r = label;
    float32* ptr_l = label_r.get_data().begin();
    int n = data.get_shape()[1];
    data = exp(data) * (1.0);
    float32* ptr = data.begin();
    for (int i = 0; i < data.get_shape()[0]; i++) {
        int idx = -1;
        float32 tmp = 0;
        for (int j = 0; j < n; j++) {
            if (*(ptr + j) > tmp) {
                idx = j;
                tmp = *(ptr + j);
            }
        }

        // std::cout << *(ptr_l) << " " << idx << endl;
        if ((int)*(ptr_l) == idx) {
            correct_n++;
        }

        ptr += n;
        ptr_l++;
    }
    cout << correct_n / (float)data.get_shape()[0] << endl;
}

void training_loop(int n_epochs, const vector<Variable<float32>>& inputs, const vector<Variable<float32>>& labels, const Variable<float32>& test_img, const Variable<float32>& test_label) {
    Model<float32> model = make_model<float32>({make_shared<Linear<float32>>(Linear<float32>(784, 256, true)),
                                                make_shared<ReLU<float32>>(),
                                                make_shared<Linear<float32>>(Linear<float32>(256, 10, true)),
                                                make_shared<LogSoftmax<float32>>()});
    // Model<float32> model = make_model<float32>({make_shared<Linear<float32>>(Linear<float32>(1, 1, true))});
    SGD<float32> optimizer(model.get_parameters(), 0.01);

    test(inputs[10], labels[10], model);

    for (int i = 0; i < n_epochs; i++) {
        Tensor<float32> total_l = {0.0};
        for (int j = 0; j < inputs.size(); j++) {
            auto prediction = model(inputs[j]);
            Variable<float32> loss = nll_loss(labels[j], prediction);
            loss.backward();
            optimizer.step();
            optimizer.zero_grads();
            total_l = total_l + loss.get_data();

            if (j % 200 == 0) {
                cout << loss.get_data() << endl;
                test(inputs[10], labels[10], model);
            }

        }

        cout << "Epoch " << i << " Loss: " << total_l / (float32)inputs.size() << endl;
    }

    test(inputs[10], labels[10], model);
}

vector<vector<float32>> Read_Boston() {
    ifstream inf;
    inf.open("../Dataset/BostonHousing/Boston_housing.txt");
    string line;

    vector<vector<float32>> _data;
    while (getline(inf, line)) {
        vector<float32> row;
        std::stringstream ss(line);
        std::string item;
        while (ss >> item) {
            row.push_back(std::stod(item));
        }

        _data.push_back(row);
    }

    inf.close();

    return _data;
}

int main() {
    // Initialize the engine and load the configurations
    NN::initializeNNEngine();

    MNIST<float32> dataset("../Dataset/MNIST/");

    auto train_data = dataset.get_train();
    auto test_data = dataset.get_test();

    auto train_list = dataset.mini_batch(128);

    cout << "Start Training" << endl;

    training_loop(50, train_list.first, train_list.second, *test_data.first, *test_data.second);

//    Model<float32> model = make_model<float32>({make_shared<Linear<float32>>(Linear<float32>(784, 256, true)),
//                                                make_shared<ReLU<float32>>(),
//                                                make_shared<Linear<float32>>(Linear<float32>(256, 10, true)),
//                                                make_shared<LogSoftmax<float32>>()});
//    SGD<float32> optimizer(model.get_parameters(), 1e-2);
//    auto input = train_list.first;
//    auto label = train_list.second;
//    for (int i = 0; i < 10; i++) {
//        auto output = model(input[0]);
//        auto loss = nll_loss(label[0], output);
//        loss.backward();
//        optimizer.step();
//        optimizer.zero_grads();
//        std::cout << loss.get_data() << std::endl;
//    }

    return 0;
}
