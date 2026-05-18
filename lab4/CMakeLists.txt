cmake_minimum_required(VERSION 3.18)
project(lab4_cuda_matrix LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(lab4
    src/main.cu
    src/matrix_file.cpp
)

target_include_directories(lab4 PRIVATE include)
set_target_properties(lab4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
