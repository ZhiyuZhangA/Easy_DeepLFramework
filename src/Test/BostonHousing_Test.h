//
// Created by zzy on 2024/8/19.
//


#include <vector>

#include <iostream>
#include "../Function/Function.h"
#include "../optimizers/Optimizer.h"
#include "../Variable/Variable.h"
#include "../Layers/Linear.h"
#include "../Layers/Tanh.h"
#include "../Layers/ReLU.h"
#include "../Model/Model.h"

using namespace std;
using namespace NN;

void training_loop(int n_epochs, const Variable<float32>& input, const Variable<float32>& label) {
    Model<float32> model = make_model<float32>({make_shared<Linear<float32>>(Linear<float32>(13, 1, true)),
    make_shared<ReLU<float32>>(),
    make_shared<Linear<float32>>(Linear<float32>(30, 13, true)),
    make_shared<Tanh<float32>>(),
    make_shared<Linear<float32>>(Linear<float32>(13, 1, true))});
    // Model<float32> model = make_model<float32>({make_shared<Linear<float32>>(Linear<float32>(1, 1, true))});
    SGD<float32> optimizer(model.get_parameters(), 1e-3);

    for (int i = 0; i < n_epochs; i++) {
        bool flag = false;
        if (i >= 10) flag = true;
        auto prediction = model(input, flag);
        //  cout << *(prediction.get_node()->_grad) << endl;
        Variable<float32> loss = mse_loss(label, prediction);
        optimizer.zero_grads();
        loss.backward();
        optimizer.step();

        if ((i % 10) == 0)
            cout << "Epoch " << i << " Loss: " << loss.get_data() << endl;

        // Validation loss
        if ((i % 10) == 0) {
            auto prediction_v = model(input, flag);
            Variable<float32> loss_V = mse_loss(label, prediction);
            cout << "Validation Loss: " << loss_V.get_data() << endl;
        }
    }
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

void Boston_Housing() {
    // Read the csv raw_data
    vector<vector<float32>> raw_data = Read_Boston();

    // Normalize the data
    for (auto& column : raw_data) {
        double minVal = *std::min_element(column.begin(), column.end());
        double maxVal = *std::max_element(column.begin(), column.end());

        for (auto& val : column) {
            val = (val - minVal) / (maxVal - minVal);
        }
    }

    // Split x and y
    vector<float32> input_r_data;
    vector<float32> label_r_data;
    for (int i = 0; i < raw_data.size(); i++) {
        for (int j = 0; j < raw_data[0].size(); j++) {
            if (j != raw_data[0].size() - 1) {
                input_r_data.push_back(raw_data[i][j]);
            }
            else {
                label_r_data.push_back(raw_data[i][j]);
            }
        }
    }

    // Create the Tensor corresponding to the raw_data
    Tensor<float32> input_data((TensorShape({raw_data.size(), raw_data[0].size() - 1})));
    Tensor<float32> labels((TensorShape({raw_data.size(), 1})));

    input_data.fill(input_r_data);
    labels.fill(label_r_data);

    cout << input_data.get_tensorShape() << endl;
    cout << labels.get_tensorShape() << endl;

    Variable<float32> var_input(input_data);
    Variable<float32> var_label(labels);

    // Start training
    training_loop(500, var_input, var_label);
}