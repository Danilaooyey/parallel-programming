cmake_minimum_required(VERSION 3.16)
project(lab3_mpi_matrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_executable(lab3
    src/main.cpp
    src/matrix_io.cpp
)

target_include_directories(lab3 PRIVATE include)
target_link_libraries(lab3 PRIVATE MPI::MPI_CXX)
