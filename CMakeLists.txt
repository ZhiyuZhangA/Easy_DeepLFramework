cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork_Base)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")

add_executable(NeuralNetwork_Base src/main.cpp
        src/Core/Tensor/tensor.h
        src/Core/Tensor/tensorBuffer.h
        src/Core/allocators/default_allocator.cpp
        src/Core/allocators/default_allocator.h
        src/Core/Base/base.h
        src/Variable/Variable.cpp
        src/Variable/Variable.h
        src/Core/Base/base.cpp
        src/utils/utils.cpp
        src/utils/utils.h
        src/Core/Tensor/tensor.cpp
        src/Function/Function.h
        src/Core/Node/base_node.h
        src/Core/autograd/autograd_graph.h
        src/Core/Node/node.h
        src/Core/Tensor/native_func_impl_t.h
        src/Function/common_func.h
        src/Function/activation_func.h
        src/Core/Tensor/activation_func_impl_t.h
        src/optimizers/Optimizer.h
        src/Function/loss_func.h
        src/Layers/LayerBase.h
        src/Layers/Linear.h
        src/Pipeline/Pipeline.h
        src/Model/Model.h
        src/Layers/Tanh.h
        src/Dataset/Dataset.h
        src/Layers/ReLU.h
        src/Dataset/MNIST.h
        src/Test/BostonHousing_Test.h
        src/Layers/Layer.h
        src/Layers/Softmax.h
        src/Core/Tensor/loss_func_impl_t.h
        src/Layers/LogSoftmax.h
)
