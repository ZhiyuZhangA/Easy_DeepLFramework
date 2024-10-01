#ifndef _TENSOR_
#define _TENSOR_

#include <cstdint>
#include <vector>
#include <stack>
#include <new>
#include <algorithm>
#include <functional>
#include <string>
#include <iomanip>
#include <ostream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <immintrin.h>
#include <omp.h>

#include "tensorBuffer.h"

namespace NN {

    // Return the shape that the tensor would broadcast to
    std::vector<size_t> broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

    // Add one to the shape while broadcasting
    std::vector<size_t> _extend_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

    // Convert nested initializer_list to vectors
    // Two dimension
    template <typename T>
    std::vector<std::vector<T>> il_to_vector(const std::initializer_list<std::initializer_list<T>>& il) {
        std::vector<std::vector<T>> res;
        for (auto d : il) {
            res.push_back(std::vector<T>(d));
        }
        return res;
    }

    template <typename T>
    // Three dimension
    std::vector<std::vector<std::vector<T>>> il_to_vector(const std::initializer_list<std::initializer_list<std::initializer_list<T>>>& il) {
        std::vector<std::vector<std::vector<T>>> res;
        for (auto i1 : il) {
            std::vector<std::vector<T>> vec;
            for (auto i2 : i1) {
                vec.push_back(std::vector<T>(i2));
            }
            res.push_back(vec);
        }
        return res;
    }

    template <typename T>
    // Four dimension
    std::vector<std::vector<std::vector<std::vector<T>>>> il_to_vector(const std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>& il) {
        std::vector<std::vector<std::vector<std::vector<T>>>> res;
        for (auto i1 : il) {
            std::vector<std::vector<std::vector<T>>> vec1;
            for (auto i2 : i1) {
                std::vector<std::vector<T>> vec2;
                for (auto i3 : i2) vec2.push_back(std::vector<T>(i3));
                vec1.push_back(vec2);
            }
            res.push_back(vec1);
        }
        return res;
    }

    struct PrintOptions {
        int precision = 0;
        int width = 0;
    };

    // Maintaining the steps and the shape of the tensor.
    class TensorShape {
    public:
        TensorShape() = default;
        TensorShape(const std::vector<size_t>& shape) : _shape(shape) { }

        inline const std::vector<size_t> get_shape() const { return _shape; }
        inline const size_t get_dim() const { return _shape.size(); }
        inline const std::vector<size_t> get_stride() const { return _stride; }
        inline void set_shape(const std::vector<size_t>& shape) { _shape = shape; }

        void update_strides() {
            _stride = get_updated_strides(_shape);
        }

        std::vector<size_t> get_updated_strides(std::vector<size_t> shape) const {
            if (shape.size() >= 2) {
                std::vector<size_t> stride;
                size_t base = 1;
                auto dim = shape.size();
                for (auto i = 1; i != dim; i++) {
                    stride.push_back(base *= shape.at(dim - i));
                }
                return stride;
            }
            return shape;
        }

        friend std::ostream &operator<<(std::ostream &os, const TensorShape &shape) {
            os << "(";
            for (int i = 0; i < shape.get_dim(); i++) {
                os << shape.get_shape()[i];
                if (i != shape.get_dim() - 1) {
                    os << ", ";
                }
            }
            os << ")";
            return os;
        }

        static size_t count_shape(const std::vector<size_t>& dim) {
            size_t cnt = 1;
            if (dim.empty())
                cnt = 0;
            else {
                for (size_t d : dim)
                    cnt *= d;
            }

            return cnt;
        }

    private:
        std::vector<size_t> _shape;
        std::vector<size_t> _shape_backup;
        std::vector<size_t> _stride;
    };

    template <typename T>
    class Tensor {
    public:
        typedef std::vector<size_t> dim_type;

        // With default initialization, the tensor would have a scalar zero
        Tensor() { }

        explicit Tensor(const TensorShape &tensorShape) : _tensorShape(tensorShape) {
            init_dim();
            _buffer = std::make_shared<tensorBuffer<T>>(sizeof(T) * _count, nullptr);
            // default constructing data of type T and filling raw memory
            default_filling_data();
            // default precision
            set_print_options(6);
        }

        // Multi-dimensional tensor
        explicit Tensor(std::initializer_list<T> data, const TensorShape &tensorShape) {
            initialize_with_shape(tensorShape.get_shape());
            fill(data);
        }

        // One-dimensional tensor
        Tensor(const std::initializer_list<T>& data) {
            initialize_with_shape({data.size()});
            fill(data);
        }

        // Two-dimensional tensor
        Tensor(const std::initializer_list<std::initializer_list<T>>& data) {
            initialize_with_shape({data.size(), data.begin()->size()});
            fill(il_to_vector(data));
        }

        // Three-dimensional tensor
        Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<T>>>& data) {
            initialize_with_shape({data.size(), data.begin()->size(), data.begin()->begin()->size()});
            fill(il_to_vector(data));
        }

        // Four-dimensional tensor
        Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>& data) {
            initialize_with_shape({data.size(), data.begin()->size(), data.begin()->begin()->size(), data.begin()->begin()->begin()->size()});
            fill(il_to_vector(data));
        }

        inline const std::vector<size_t> get_shape() const { return _tensorShape.get_shape(); }
        inline const TensorShape get_tensorShape() const { return _tensorShape; }
        inline const std::vector<size_t> get_stride() const { return _tensorShape.get_stride(); }
        inline const size_t get_dim() const { return _tensorShape.get_dim(); }
        inline const size_t size() const { return _count; }
        inline T* begin() const { return _buffer->data_ptr_t(); }
        inline T* end() const { return _buffer->data_ptr_t() + _count; }

    public:
        static Tensor<T> ones(std::vector<size_t> shape) {
            Tensor<T> res((TensorShape(shape)));
            res.fill(1);
            return res;
        }

        void zeros() {
            if (_buffer == nullptr || _buffer->data_ptr_t() == nullptr || _buffer->rdata() == nullptr)
                throw std::runtime_error("Using uninitialized memory for the tensor");

            fill(0);
        }

        void ones() {
            if (_buffer == nullptr || _buffer->data_ptr_t() == nullptr || _buffer->rdata() == nullptr)
                throw std::runtime_error("Using uninitialized memory for the tensor");

            fill(1);
        }

        std::vector<T> to_std_vector() {
            return (std::vector<T>(begin(), end()));
        }

        void fill(T data) {
            for (T* ptr = begin(); ptr != end(); ptr++)
                *ptr = data;
        }

        // 1-d tensor or multidimensional tensor.
        bool fill(const std::vector<T>& data) {
            if (data.size() != _count) {
                std::cerr << "Filling tensor error: filled array doesn't have the same number of elements as the tensor!!!" << std::endl;
                return false;
            }

            int idx = 0;
            for (T* ptr = begin(); ptr != end(); ptr++) *ptr = data[idx++];
            return true;
        }

        // 2-d tensor - Matrix
        bool fill(const std::vector<std::vector<T>>& data) {
            // whether dimension is 2d
            if (_tensorShape.get_dim() != 2) {
                std::cerr << "Filling tensor error: filled array is 2D, but the tensor is " << _tensorShape.get_dim() << "D!" << std::endl;
                return false;
            }

            // whether the shape is the same
            if (_tensorShape.get_shape()[0] != data.size() || _tensorShape.get_shape()[1] != data[0].size()) {
                std::cerr << "Filling tensor error: cannot fill tensor of shape " << shape_to_string(
                        _tensorShape.get_shape())
                << " with array of shape (" << std::to_string(data.size()) + ", " + std::to_string(data[0].size()) + ")" << std::endl;
                return false;
            }

            auto ncol = data[0].size();
            for (int i = 0; i < data.size(); i++) {
                if (ncol != data[i].size()) {
                    std::cerr << "Filled array doesn't have an agreed shape in column number!" << std::endl;
                    return false;
                }
            }

            T* ptr = begin();
            for (int i = 0; i < data.size(); i++)
                for (int j = 0; j < data[0].size(); j++)  *(ptr++) = data[i][j];

            return true;
        }

        // 3-d Tensor
        bool fill(const std::vector<std::vector<std::vector<T>>>& data) {
            // whether dimension is 3d
            if (_tensorShape.get_dim() != 3) {
                std::cerr << "Filling tensor error: filled array is 3D, but the tensor is " << _tensorShape.get_dim() << "D!" << std::endl;
                return false;
            }

            // whether the shape is the same
            if (_tensorShape.get_shape()[0] != data.size() || _tensorShape.get_shape()[1] != data[0].size() ||
                    _tensorShape.get_shape()[2] != data[0][0].size()) {
                std::cerr << "Filling tensor error: cannot fill tensor of shape " << shape_to_string(
                        _tensorShape.get_shape())
                          << " with array of shape (" << std::to_string(data.size()) + ", " + std::to_string(data[0].size()) + ", " + std::to_string(data[0][0].size()) + ")" << std::endl;
                return false;
            }

            auto nrow = data[0].size();
            auto ncol = data[0][0].size();
            for (int i = 0; i < data.size(); i++) {
                if (nrow != data[i].size()) {
                    std::cerr << "Filled array doesn't have an agreed shape in row number (0, 1, 0)!" << std::endl;
                    return false;
                }
                for (int j = 0; j < data[i].size(); j++) {
                    if (ncol != data[i][j].size()) {
                        std::cerr << "Filled array doesn't have an agreed shape in column number (0, 0, 1)!" << std::endl;
                        return false;
                    }
                }
            }

            T* ptr = begin();
            for (int i = 0; i < data.size(); i++) {
                for (int j = 0; j < data[0].size(); j++) {
                    for (int k = 0; k < data[0][0].size(); k++) {
                        *(ptr++) = data[i][j][k];
                    }
                }
            }

            return true;
        }

        // 4-d Tensor
        bool fill(const std::vector<std::vector<std::vector<std::vector<T>>>>& data) {
            // whether dimension is 4d
            if (_tensorShape.get_dim() != 4) {
                std::cerr << "Filling tensor error: filled array is 4D, but the tensor is " << _tensorShape.get_dim() << "D!" << std::endl;
                return false;
            }

            // whether the shape is the same
            if (_tensorShape.get_shape()[0] != data.size() || _tensorShape.get_shape()[1] != data[0].size() ||
                    _tensorShape.get_shape()[2] != data[0][0].size() ||
                    _tensorShape.get_shape()[3] != data[0][0][0].size()) {
                std::cerr << "Filling tensor error: cannot fill tensor of shape " << shape_to_string(
                        _tensorShape.get_shape())
                          << " with array of shape (" << std::to_string(data.size()) + ", " + std::to_string(data[0].size()) + ", " + std::to_string(data[0][0].size()) + ", " + std::to_string(data[0][0][0].size()) + ")" << std::endl;
                return false;
            }

            auto ncha = data[0].size();
            auto nrow = data[0][0].size();
            auto ncol = data[0][0][0].size();
            for (int i = 0; i < data.size(); i++) {
                if (ncha != data[i].size()) {
                    std::cerr << "Filled array doesn't have an agreed shape in channel number (0, 1, 0, 0)!" << std::endl;
                    return false;
                }
                for (int j = 0; j < data[i].size(); j++) {
                    if (nrow != data[i][j].size()) {
                        std::cerr << "Filled array doesn't have an agreed shape in row number (0, 0, 1, 0)!" << std::endl;
                        return false;
                    }
                    for (int k = 0; k < data[i][j].size(); k++) {
                        if (ncol != data[i][j][k].size()) {
                            std::cerr << "Filled array doesn't have an agreed shape in column number (0, 0, 0, 1)!" << std::endl;
                            return false;
                        }
                    }
                }
            }

            T* ptr = begin();
            for (int i = 0; i < data.size(); i++) {
                for (int j = 0; j < data[0].size(); j++) {
                    for (int k = 0; k < data[0][0].size(); k++) {
                        for (int l = 0; l < data[0][0][0].size(); l++) *(ptr++) = data[i][j][k][l];
                    }
                }
            }

            return true;
        }

        T sum() {
            T sum = 0;
            for (T* ptr = begin(); ptr != end(); ptr++) {
                sum += *ptr;
            }
            return sum;
        }

        T& at(const std::vector<size_t>& pos) {
            if (pos.size() != _tensorShape.get_dim())
                throw std::invalid_argument("Dimension of given index doesn't confirm with the dimension of the tensor");

            T* ptr = begin();
            auto offset = 0;
            for (int i = 0; i < pos.size() - 1; i++) {
                offset += pos[i] * _tensorShape.get_stride()[_tensorShape.get_stride().size() - i - 1];
            }
            offset += pos[pos.size() - 1];

            return *(ptr + offset);
        }

        void reshape(const std::vector<size_t>& dim) {
            // Check the validity of dimension provided.
            size_t cnt = TensorShape::count_shape(dim);

            if (cnt != _count) {
                std::cerr << "cannot reshape tensor of size " << _count << " into shape " << shape_to_string(dim);
                return;
            }

            // Set the dim and stride
            _tensorShape.set_shape(dim);
            _tensorShape.update_strides();
        }

        Tensor<T> transpose() {
            /* Reverse the shape (2,3,4) -> (4,3,2)
             * Iterate through the entire tensor, then compute the index for every data
             * Reverse the index vector, then assign it to the result tensor.
             * */
            std::vector<size_t> res_shape = get_shape();
            std::reverse(res_shape.begin(), res_shape.end());
            Tensor<T> res_tensor((TensorShape(res_shape)));

            T* ptr = begin();
            std::vector<size_t> indices(this->get_dim(), 0);
            for (auto i = 0; i < this->size(); i++) {
                // Get the corresponding indices in the original tensor
                size_t remaining = i;
                for (int j = this->get_dim() - 1; j >= 0; j--) {
                    indices[j] = remaining % get_shape()[j];
                    remaining /= get_shape()[j];
                }

                // reverse the indices
                std::reverse(indices.begin(), indices.end());
                res_tensor.at(indices) = *(ptr + i);
            }

            return res_tensor;
        }

        Tensor<T> sum_to(const std::vector<size_t>& tar_shape) {
            // Extend the shape if the dimension doesn't match
            auto res_shape = tar_shape;
            if (this->get_dim() != tar_shape.size()) {
                res_shape = _extend_shape(tar_shape, this->get_shape());
            }

            // Defining the result Tensor with extending shape
            Tensor<T> res_tensor((TensorShape(res_shape)));

            std::vector<size_t> indices(this->get_dim(), 0);
            std::vector<T> filled_data(size());

            for (size_t i = 0; i < size(); i++) {
                // Get the corresponding indices in the original tensor
                size_t remaining = i;
                for (int j = this->get_dim() - 1; j >= 0; j--) {
                    indices[j] = remaining % get_shape()[j];
                    remaining /= get_shape()[j];
                }

                std::vector<size_t> target_indices(res_shape.size());
                for (size_t j = 0; j < res_shape.size(); j++) {
                    target_indices[j] = (res_shape[j] == get_shape()[j]) ? indices[j] : 0;
                }

                res_tensor.at(target_indices) += *(begin() + i);
            }

            res_tensor.reshape(tar_shape);
            return res_tensor;
        }

//        Tensor<T> sum_to(const std::vector<size_t>& tar_shape) {
//            Tensor<T> res_data(tar_shape);
//            std::vector<size_t> res_shape = tar_shape;
//            if (this->get_dim() != tar_shape.size()) {
//                res_shape = _extend_shape(tar_shape, this->get_shape());
//            }
//
//            std::vector<T> iter_vec(this->size());
//            T* filled_ptr = begin();
//            for (int i = 0; i < this->size(); i++) {
//                iter_vec[i] = *filled_ptr;
//                filled_ptr++;
//            }
//
//            for (int i = 0; i < this->get_dim(); i++) {
//                if (res_shape[i] != this->get_shape()[i]) {
//                    if (i != this->get_dim() - 1) {
//                        int stride = this->get_stride()[this->get_dim() - 1 - i - 1];
//                        // Result tensor ptr declaration
//                        std::vector<T> tmp_vec(stride);
//                        int vec_idx = 0;
//                        // k is the index for the iter_vec
//                        for (int k = 0; k < stride; k++) {
//                            for (int j = 0; j < (this->get_shape()[i] - res_shape[i]); j++) {
//                                tmp_vec[vec_idx] = iter_vec[k] + iter_vec[k + (j + 1) * stride];
//                            }
//                            vec_idx++;
//                        }
//                        iter_vec = tmp_vec;
//                    }
//                    else {
//
//                    }
//                }
//            }
//
//            res_data.fill(iter_vec);
//            return res_data;
//        }

        friend Tensor<T> broadcast_data(const Tensor<T>& c1, const Tensor<T>& c2, const std::function<T(T&, T&)>& op) {
            std::vector<size_t> res_shape = broadcast_shape(c1.get_shape(), c2.get_shape());
            Tensor<T> res_tensor((TensorShape(res_shape)));
            // Extend the shape with one
            std::vector<size_t> c1_stride = c1.get_stride();
            std::vector<size_t> c2_stride = c2.get_stride();
            if (c1.get_dim() != c2.get_dim()) {
                // Find the tensor with smaller dimension
                std::vector<size_t> tmp_shape = _extend_shape(c1.get_shape(), c2.get_shape());
                // Update temporary stride
                if (c1.get_dim() != tmp_shape.size()) {
                    c1_stride = c1._tensorShape.get_updated_strides(tmp_shape);
                }
                else {
                    c2_stride = c2._tensorShape.get_updated_strides(tmp_shape);
                }
            }

            int cnt = res_tensor.size();
            std::vector<T> vec(cnt);

            T* ptr1 = c1.begin();
            T* ptr2 = c2.begin();
            int offset = 1;
            int step_cnt = 1;
            bool flag = 0;
            if (!c1_stride.empty() && !c2_stride.empty()) {
                offset = std::max(c1_stride[0], c2_stride[0]) / std::min(c1_stride[0], c2_stride[0]);
                flag = 1;
            }

            // Filling data using circular pointer for both tensor
            for (int i = 0; i < cnt; i++) {
                if (op)
                    vec[i] = op(*ptr1, *ptr2);
                else { // if operation is not given, then simply fill the vector with the smallest tensor data
                    if (c1.size() < c2.size())
                        vec[i] = *ptr1;
                    else
                        vec[i] = *ptr2;
                }

                if (flag) {
                    step_cnt++;
                    if (c1_stride[0] > c2_stride[0]) {
                        ptr1 = c1._loop_ptr(ptr1);
                        if ((i + 1) % offset == 0)
                            ptr2 = c2._loop_ptr(ptr2);
                    }
                    else {
                        ptr2 = c2._loop_ptr(ptr2);
                        if ((i + 1) % offset == 0)
                            ptr1 = c1._loop_ptr(ptr1);
                    }
                }
            }

            res_tensor.fill(vec);
            return res_tensor;
        }

        friend bool operator==(const Tensor<T>& c1, const Tensor<T>& c2) {
            // Same dimension
            if (c1._tensorShape.get_dim() != c2._tensorShape.get_dim()) return false;

            // Same shape
            for (int i = 0; i < c1._tensorShape.get_dim(); i++) {
                if (c1._tensorShape.get_shape()[i] != c2._tensorShape.get_shape()[i]) return false;
            }

            // Same data
            T* ptr1 = c1.begin();
            T* ptr2 = c2.begin();
            for (int i = 0; i < c1._count; i++) {
                if (*(ptr1++) != *(ptr2++)) return false;
            }

            return true;
        }

        Tensor<T> operator+(const Tensor<T>& other) {
            // Same dimension
            if (this->get_dim() == other.get_dim()) {
                // Same shape
                if (this->get_shape() == other.get_shape()) {
                    return add_op(other, true);
                }
            }

            // broadcasting
            return broadcast_data(*this, other, std::function<T(T&, T&)>([](T& a, T& b) -> T {
                return a + b;
            }));
        }

        Tensor<T> operator+(const T& other) {
            Tensor<T> res(this->get_tensorShape());
            T* ptr_r = res.begin();
            for (T* ptr = this->begin(); ptr != this->end(); ptr++) {
                *ptr_r = *ptr + other;
                ptr_r++;
            }
            return res;
        }

        Tensor<T> operator-(const Tensor<T>& other) {
            // Same dimension
            if (this->get_dim() == other.get_dim()) {
                // Same shape
                if (this->get_shape() == other.get_shape()) {
                    return sub_op(other, true);
                }
            }

            // broadcasting
            return broadcast_data(*this, other, std::function<T(T&, T&)>([](T& a, T& b) -> T {
                return a - b;
            }));
        }

        Tensor<T> operator-(const T& other) {
            Tensor<T> res(this->get_tensorShape());
            T* ptr_r = res.begin();
            for (T* ptr = this->begin(); ptr != this->end(); ptr++) {
                *ptr_r = *ptr - other;
                ptr_r++;
            }
            return res;
        }

        friend Tensor<T> operator-(const T& var1, const Tensor<T>& var2) {
            Tensor<T> res((TensorShape(var2.get_shape())));
            T* ptr_r = res.begin();
            for (T* ptr = var2.begin(); ptr != var2.end(); ptr++) {
                *ptr_r = 1 - *ptr;
                ptr_r++;
            }
            return res;
        }

        Tensor<T> operator/(const Tensor<T>& other) {
            // Same dimension
            if (this->get_dim() == other.get_dim()) {
                // Same shape
                if (this->get_shape() == other.get_shape()) {
                    return div_op(other, true);
                }
            }

            // broadcasting
            return broadcast_data(*this, other, std::function<T(T&, T&)>([](T& a, T& b) -> T {
                return a / b;
            }));
        }

        Tensor<T> operator/(const T& other) {
            assert(other != 0);
            Tensor<T> res(this->get_tensorShape());
            T* ptr_r = res.begin();
            for (T* ptr = this->begin(); ptr != this->end(); ptr++) {
                *ptr_r = *ptr / other;
                ptr_r++;
            }
            return res;
        }

        Tensor<T> operator*(const Tensor<T>& other) {
            // Same dimension
            if (this->get_dim() == other.get_dim()) {
                // Same shape
                if (this->get_shape() == other.get_shape()) {
                    return mul_op(other, true);
                }
            }

            // broadcasting
            return broadcast_data(*this, other, std::function<T(T&, T&)>([](T& a, T& b) -> T {
                return a * b;
            }));
        }

        Tensor<T> operator*(const T& other) {
            Tensor<T> res(this->get_tensorShape());
            T* ptr_r = res.begin();
            for (T* ptr = this->begin(); ptr != this->end(); ptr++) {
                *ptr_r = *ptr * other;
                ptr_r++;
            }
            return res;
        }

        Tensor<T>& operator+=(const Tensor<T>& other) {
            *this = *this + other;
            return *this;
        }

        Tensor<T>& operator-=(const Tensor<T>& other) {
            *this = *this - other;
            return *this;
        }

        Tensor<T>& operator*=(const Tensor<T>& other) {
            *this = *this * other;
            return *this;
        }

        Tensor<T>& operator/=(const Tensor<T>& other) {
            *this = *this / other;
            return *this;
        }

//        Tensor<T> matmul(const Tensor<T>& other);
//
//        template <>
        Tensor<float32> matmul(const Tensor<float32>& other) {
            // 无广播
            assert(this->get_shape().back() == other.get_shape()[other.get_dim() - 2]);

            const size_t BLOCK_SIZE = 32;
            auto N_i = get_shape()[get_dim() - 2];
            auto N_j = other.get_shape().back();
            auto N_k = get_shape().back();

            // Defining the result Tensor
            std::vector<size_t> res_shape(this->get_shape());
            res_shape.back() = other.get_shape().back();
            Tensor<float32> result((TensorShape(res_shape)));

            // Defining the pointer to the tensors
            auto res_ptr = result.begin();
            auto t1_ptr = begin();
            auto t2_ptr = other.begin();

            // remaining size
            size_t m_size1 = N_k * N_i;
            size_t m_size2 = N_j * N_k;
            size_t m_size_res = N_i * N_j;
            size_t remain_s = this->size() / m_size1;
            float sum_arr[8];
            for (int n = 0; n < remain_s; n++) {
                for (auto bi = 0; bi < N_i; bi+=BLOCK_SIZE) {
                    // other->col
                    for (auto bj = 0; bj < N_j; bj+=BLOCK_SIZE) {
                        auto i_max = std::min((bi + BLOCK_SIZE), N_i);
                        auto j_max = std::min(bj + BLOCK_SIZE, N_j);

                        for (auto i = bi; i < i_max; i++) {
                            for (auto j = bj; j < j_max; j++) {
                                auto k = 0;
                                __m256 sum = _mm256_setzero_ps();
                                for (; k + 7 < N_k; k+=8) {
                                    __m256 a = _mm256_load_ps(t1_ptr + i * N_k + k);
                                    __m256 b = _mm256_load_ps(t2_ptr + k * N_j + j);
                                    sum = _mm256_fmadd_ps(a, b, sum);
                                }

                                float total_sum = 0;
                                sum = _mm256_hadd_ps(sum, sum);
                                sum = _mm256_hadd_ps(sum, sum);
                                _mm256_store_ps(sum_arr, sum);
                                total_sum += sum_arr[0] + sum_arr[4];

                                for (; k < N_k; ++k) {
                                    total_sum += *(t1_ptr + i * N_k + k) * *(t2_ptr + k * N_j + j);
                                }

                                *res_ptr = total_sum;
                                res_ptr++;
                            }
                        }
                    }
                }
                // res_ptr += m_size_res;
                t1_ptr += m_size1;
                t2_ptr += m_size2;
            }


//            int cnt = 0; // 65536
            // 多少个内层matrix
//            for (int n = 0; n < remain_s; n++) {
//                // this->row
//
////                #pragma omp parallel for
//                for (int i = 0; i < N_i; i+=4) {
//                    for (int j = 0; j < N_j; j+=2) {
//
//                        __m256 acc[4] = {};
//                        for (int k = 0; k < N_k; k++) {
//                            for (int iy = 0; iy < 4; iy++) {
//                                __m256 ta = _mm256_broadcast_ss(t1_ptr + (i + iy) * N_i + k);
//                                __m256 b = _mm256_load_ps(t2_ptr + (j * N_j + k * 8) / 8);
//                                acc[iy] = _mm256_fmadd_ps(ta, b, acc[iy]);
//                            }
//                        }
//
//                        for (int iy = 0; iy < 4; iy++) {
//                            float tmp[8];
//                            _mm256_store_ps(tmp, acc[iy]);
//                            for (int z = 0; z < 8; z++) {
//                                *(res_ptr + ((i + iy) * N_i + j) / 8 + z)  = tmp[z];
//                            }
//                        }
//                    }
//                }
//
////                for (int i = 0; i < N_i; i++) {
////                     for (int j = 0; j < N_j; j++) {
////                         for (int k = 0; k < N_k; k++) {
////                            *res_ptr = *(t1_ptr + i * N_k + k) * *(t2_ptr + k * N_j + j);
////                        }
////                        res_ptr++;
////                    }
////                }
//
//                //#pragma omp parallel for
////                for (int i = 0; i < N_i; i++) {
////                    // other->col
////                    for (int j = 0; j < N_j; j++) {
////                        __m256 sum = _mm256_setzero_ps();
////                        // this->col
////                        int k = 0;
////                        for (; k + 7 < N_k; k+=8) {
////                            // Compute Linear indices
////                            __m256 a = _mm256_load_ps(t1_ptr + i * N_k + k);
////                            __m256 b = _mm256_load_ps(t2_ptr + k * N_j + j);
////                            sum = _mm256_fmadd_ps(a, b, sum);
////                        }
////
////                        float total_sum = 0;
////                        float sum_arr[8];
////                        _mm256_store_ps(sum_arr, sum);
////                        for (int l = 0; l < 8; ++l) {
////                            total_sum += sum_arr[l];
////                        }
////
////                        // remaining element
////                        for (; k < N_k; ++k) {
////                            total_sum += *(t1_ptr + i * N_k + k) * *(t2_ptr + k * N_j + j);
////                        }
////
////                        *res_ptr = total_sum;
////                        res_ptr++;
////                    }
////                }
//                res_ptr += m_size_res;
//                t1_ptr += m_size1;
//                t2_ptr += m_size2;
//            }

            return result;
        }

        Tensor<T> mean() {
            Tensor<T> res({0});
            T sum = 0;
            for (T* ptr = this->begin(); ptr != this->end(); ptr++) {
                sum += *ptr;
            }
            sum /= this->_count;
            res.fill(sum);
            return res;
        }

        void set_print_options(const size_t& precision) {
            _printOptions.precision = precision;
        }

        friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
            T* ptr = tensor.begin();

            if (ptr == nullptr) {
                os << "tensor([])\n";
                return os;
            }

            std::stack<char> paren_s; // stack for medium parentheses
            int g_cnt = 1;

            // Set format of the floating output
            os << std::fixed << std::setprecision(_printOptions.precision);

            T max_val = std::floor(*std::max_element(ptr, ptr + tensor._count));
            _printOptions.width = std::to_string(static_cast<long>(max_val)).length() + 1;

            if constexpr (std::is_floating_point<T>::value) _printOptions.width += _printOptions.precision + 1;

            os << "tensor(";

            for (int i = 0; i < tensor._tensorShape.get_dim(); i++) {
                paren_s.push('[');
                os << '[';
            } // After finish, the buffer would be like [[[...

            while (ptr != tensor.end()) {
                int tmp_cnt = 1; // 对每个部分计数
                for (int i = 0; i < tensor._count / tensor._tensorShape.get_shape()[0]; i++) {
                    os << std::setw(Tensor::_printOptions.width) << *ptr;
                    // After lines
                    if (!tensor._tensorShape.get_stride().empty() && tmp_cnt % tensor._tensorShape.get_stride()[0] == 0) {
                        int removed_brac = 0;
                        for (size_t j : tensor._tensorShape.get_stride()) {
                            if (tmp_cnt < j || tmp_cnt % j != 0)
                                break;
                            os << std::string("]");
                            paren_s.pop();
                            removed_brac++;
                        }
                        // 设置换行
                        if (!(g_cnt + 1 > tensor._count)) {
                            os << ",\n";
                            if (removed_brac > 1) os << "\n";
                            // Compensate the spaces before each middle parenthesis.
                            for (int r = 0; r < paren_s.size() + 7; r++) os << " ";
                            // Add middle parenthesis to the next line.
                            for (int c = 0; c < removed_brac; c++) {
                                os << "[";
                                paren_s.push('[');
                            }
                        }
                    }
                    else {
                        if (!(tensor._tensorShape.get_dim() == 1 && g_cnt >= tensor._count))
                            os << ", ";
                    }

                    ptr++;
                    tmp_cnt++;
                    g_cnt++;
                }
            }

            while(!paren_s.empty()) {
                os << "]";
                paren_s.pop();
            }

            os << ")\n";
            return os;
        }

    private:
        void init_dim() {
            // assign the count of the tensor
            _count = TensorShape::count_shape(_tensorShape.get_shape());
            // set the step of the tensor
            _tensorShape.update_strides();
        }

        void default_filling_data(bool construct=true) {
            if (construct || !_buffer->data_ptr_t()) {
                _buffer->data_ptr_t() = new(_buffer->rdata()) T[_count]();
            }
            else {
                zeros();
            }
        }

        std::string shape_to_string(const std::vector<size_t> shape) {
            std::string buffer;
            buffer.push_back('(');
            for (int i = 0; i < shape.size(); i++) {
                buffer.append(std::to_string(shape[i]));
                if (i != shape.size() - 1)
                    buffer.append(", ");
            }
            buffer.push_back(')');
            return buffer;
        }

        // Increment the pointer to the raw memory in loop
        T* _loop_ptr(T* ptr) const {
            if (ptr == nullptr)
                throw std::invalid_argument("Argument provided for next_ptr operation is nullptr!");

            ptr++;
            if (ptr == end())
                ptr = begin();
            return ptr;
        }

        Tensor<T> add_op(const Tensor<T>& other, bool broadcast=false) {
            if (!broadcast)
                assert(this->get_shape() == other.get_shape());

            T* ptr1 = this->begin();
            T* ptr2 = other.begin();
            Tensor<T> res_tensor(this->get_tensorShape());
            std::vector<T> res_data;
            while (ptr1 != this->end() && ptr2 != other.end()) {
                res_data.push_back(*ptr1 + *ptr2);
                ptr1++;
                ptr2++;
            }
            res_tensor.fill(res_data);
            return res_tensor;
        }

        Tensor<T> sub_op(const Tensor<T>& other, bool broadcast=false) {
            if (!broadcast)
                assert(this->get_shape() == other.get_shape());

            T* ptr1 = this->begin();
            T* ptr2 = other.begin();
            Tensor<T> res_tensor(this->get_tensorShape());
            std::vector<T> res_data;
            while (ptr1 != this->end() && ptr2 != other.end()) {
                res_data.push_back(*ptr1 - *ptr2);
                ptr1++;
                ptr2++;
            }
            res_tensor.fill(res_data);
            return res_tensor;
        }

        Tensor<T> mul_op(const Tensor<T>& other, bool broadcast=false) {
            if (!broadcast)
                assert(this->get_shape() == other.get_shape());

            T* ptr1 = this->begin();
            T* ptr2 = other.begin();
            Tensor<T> res_tensor(this->get_tensorShape());
            std::vector<T> res_data;
            while (ptr1 != this->end() && ptr2 != other.end()) {
                res_data.push_back(*ptr1 * *ptr2);
                ptr1++;
                ptr2++;
            }
            res_tensor.fill(res_data);
            return res_tensor;
        }

        Tensor<T> div_op(const Tensor<T>& other, bool broadcast=false) {
            if (!broadcast)
                assert(this->get_shape() == other.get_shape());

            T* ptr1 = this->begin();
            T* ptr2 = other.begin();
            Tensor<T> res_tensor(this->get_tensorShape());
            std::vector<T> res_data;
            while (ptr1 != this->end() && ptr2 != other.end()) {
                res_data.push_back(*ptr1 / *ptr2);
                ptr1++;
                ptr2++;
            }
            res_tensor.fill(res_data);
            return res_tensor;
        }

        void initialize_with_shape(const std::vector<size_t>& shape) {
            _tensorShape.set_shape(shape);
            init_dim();
            _buffer = std::make_shared<tensorBuffer<T>>(sizeof(T) * _count, nullptr);
            // default constructing data of type T and filling raw memory
            default_filling_data();
            // default precision
            set_print_options(4);
        }

    private:
        std::shared_ptr<tensorBuffer<T>> _buffer;
        size_t _count; // number of element in the tensor.
        TensorShape _tensorShape;
        static PrintOptions _printOptions;
    };


    template <typename T>
    PrintOptions Tensor<T>::_printOptions;

    template <typename T>
    Tensor<T> arange(T stop, T start=0, T step=1) {
        TensorShape shape({static_cast<size_t>((stop - start) / step)});
        Tensor<T> tensor(shape);
        T* ptr = tensor.begin();
        for (int i = 0; i < tensor.size(); i+=step) {
            *(ptr++) = i;
        }

        return tensor;
    }

    template <typename T>
    std::vector<T> arange_vec(T stop, T start=0, T step=1) {
        std::vector<T> vec((stop - start) / step);
        T _data = start;
        for (int i = 0; i < vec.size(); i++) {
            vec[i] = _data;
            _data += step;
        }

        return vec;
    }



//    template <>
//    Tensor<float64> Tensor<float64>::matmul(const Tensor<float64>& other) {
//
//    }
}


#endif
