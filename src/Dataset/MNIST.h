//
// Created by zzy on 2024/8/18.
//

#ifndef NEURALNETWORK_BASE_MNIST_H
#define NEURALNETWORK_BASE_MNIST_H

#include "../Variable/Variable.h"
#include <utility>

namespace NN {
    template <typename T>
    class MNIST {
    public:
        explicit MNIST(std::string data_path) : _data_path(data_path) {
            LoadData(data_path);
        }

        void one_hot() {

        }

        std::pair<std::vector<Variable<T>>, std::vector<Variable<T>>> mini_batch(int batch_size) {
            int batch_count = 60000 / batch_size;
            std::vector<Variable<T>> train_imgs;
            std::vector<Variable<T>> train_labels;
            std::vector<T> tmp_data;
            std::vector<T> tmp_label;
            T* ptr = train_imgs_ptr->get_data().begin();
            T* ptr_l = train_labels_ptr->get_data().begin();
            for (int i = 0; i < batch_count; i++) {
                Tensor<T> tensor((TensorShape({static_cast<unsigned long long>(batch_size), 28*28})));
                tmp_data = std::vector<T>(ptr, ptr + batch_size * 28 * 28);
                tensor.fill(tmp_data);
                train_imgs.push_back((Variable<T>(tensor)));

                Tensor<T> label((TensorShape({static_cast<unsigned long long>(batch_size), 1})));
                tmp_label = std::vector<T>(ptr_l, ptr_l + batch_size);
                label.fill(tmp_label);
                train_labels.push_back((Variable<T>(label)));

                tmp_data.clear();
                tmp_label.clear();

                ptr += batch_size * 28 * 28;
                ptr_l += batch_size;
            }

            return std::pair(train_imgs, train_labels);
        }

        std::pair<std::shared_ptr<Variable<T>>, std::shared_ptr<Variable<T>>> get_train() {
            return std::pair(train_imgs_ptr, train_labels_ptr);
        }

        std::pair<std::shared_ptr<Variable<T>>, std::shared_ptr<Variable<T>>> get_test() {
            return std::pair(test_imgs_ptr, test_labels_ptr);
        }

    private:
        void LoadData(std::string data_path) {
            test_imgs_ptr = std::make_shared<Variable<T>>(read_data(data_path + test_imgs_path));
            test_labels_ptr = std::make_shared<Variable<T>>(read_label(data_path + test_labels_path));
            train_imgs_ptr = std::make_shared<Variable<T>>(read_data(data_path + train_imgs_path));
            train_labels_ptr = std::make_shared<Variable<T>>(read_label(data_path + train_labels_path));
        }

        int ReverseInt(unsigned char a, unsigned char b, unsigned char c, unsigned char d)
        {
            return ((((a * 256) + b) * 256) + c) * 256 + d;
        }

        Variable<T> read_data(std::string path) {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Unable to open the file: " + path);
            }

            unsigned char a, b, c, d;
            file >> a >> b >> c >> d;
            int magic_num = ReverseInt(a, b, c, d);
            file >> a >> b >> c >> d;
            int num_imgs = ReverseInt(a, b, c, d);
            file >> a >> b >> c >> d;
            int n_rows = ReverseInt(a, b, c, d);
            file >> a >> b >> c >> d;
            int n_cols = ReverseInt(a, b, c, d);

            Tensor<T> tensor((TensorShape({static_cast<size_t>(num_imgs), static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols)})));
            std::vector<T> data;
            for (int i = 0; i < num_imgs; i++) {
                for (int row = 0; row < n_rows; row++) {
                    for (int col = 0; col < n_cols; col++) {
                        unsigned char temp=0;
                        file.read((char*)&temp,sizeof(temp));
                        data.push_back(((T)temp) / 255);
                    }
                }
            }

            file.close();

            tensor.fill(data);
            Variable<T> var(tensor);
            return var;
        }

        Variable<T> read_label(std::string path) {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Unable to open the file: " + path);
            }

            unsigned char a, b, c, d;
            file >> a >> b >> c >> d;
            int magic_num = ReverseInt(a, b, c, d);
            file >> a >> b >> c >> d;
            int num_imgs = ReverseInt(a, b, c, d);

            Tensor<T> tensor((TensorShape({static_cast<size_t>(num_imgs), 1})));
            std::vector<T> data;
            for (int i = 0; i < num_imgs; i++) {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                data.push_back((T)temp);
            }

            file.close();

            tensor.fill(data);
            Variable<T> var(tensor);
            return var;
        }

    private:
        std::string _data_path;
        const std::string train_imgs_path = "train-images-idx3-ubyte.gz";
        const std::string test_imgs_path = "t10k-images-idx3-ubyte.gz";
        const std::string train_labels_path = "train-labels-idx1-ubyte.gz";
        const std::string test_labels_path = "t10k-labels-idx1-ubyte.gz";

        std::shared_ptr<Variable<T>> test_imgs_ptr;
        std::shared_ptr<Variable<T>> test_labels_ptr;
        std::shared_ptr<Variable<T>> train_imgs_ptr;
        std::shared_ptr<Variable<T>> train_labels_ptr;
    };
}



#endif //NEURALNETWORK_BASE_MNIST_H
