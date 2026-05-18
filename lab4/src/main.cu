#include "matrix_file.hpp"

#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct device_buffer {
    double* data = nullptr;

    explicit device_buffer(std::size_t bytes) {
        cudaError_t code = cudaMalloc(&data, bytes);
        if (code != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(code));
        }
    }

    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    ~device_buffer() {
        if (data != nullptr) {
            cudaFree(data);
        }
    }
};

void check_cuda(cudaError_t code) {
    if (code != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(code));
    }
}

struct options {
    std::string left_path = "matrix1.txt";
    std::string right_path = "matrix2.txt";
    std::string out_path = "matrixResult.txt";
    int tile = 16;
};

options parse_options(int argc, char** argv) {
    options result;
    if (argc > 1) {
        result.left_path = argv[1];
    }
    if (argc > 2) {
        result.right_path = argv[2];
    }
    if (argc > 3) {
        result.out_path = argv[3];
    }
    if (argc > 4) {
        result.tile = std::stoi(argv[4]);
    }
    if (result.tile <= 0 || result.tile > 32) {
        throw std::runtime_error("tile must be in range 1..32");
    }
    return result;
}

__global__ void tiled_product(const double* left, const double* right, double* out, int order, int tile) {
    extern __shared__ double cache[];
    double* left_tile = cache;
    double* right_tile = cache + tile * tile;

    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;
    const int row = blockIdx.y * tile + local_y;
    const int column = blockIdx.x * tile + local_x;
    double value = 0.0;

    for (int base = 0; base < order; base += tile) {
        const int a_column = base + local_x;
        const int b_row = base + local_y;
        left_tile[local_y * tile + local_x] = row < order && a_column < order ? left[row * order + a_column] : 0.0;
        right_tile[local_y * tile + local_x] = b_row < order && column < order ? right[b_row * order + column] : 0.0;
        __syncthreads();

        int k = 0;
        while (k < tile) {
            value += left_tile[local_y * tile + k] * right_tile[k * tile + local_x];
            ++k;
        }
        __syncthreads();
    }

    if (row < order && column < order) {
        out[row * order + column] = value;
    }
}

}

int main(int argc, char** argv) {
    try {
        const options opts = parse_options(argc, argv);
        lab::square_matrix left = lab::read_square_matrix(opts.left_path);
        lab::square_matrix right = lab::read_square_matrix(opts.right_path);
        if (left.order != right.order) {
            throw std::runtime_error("matrix orders are different");
        }

        lab::square_matrix answer;
        answer.order = left.order;
        answer.cells.assign(static_cast<std::size_t>(answer.order) * answer.order, 0.0);

        const std::size_t bytes = answer.cells.size() * sizeof(double);
        device_buffer gpu_left(bytes);
        device_buffer gpu_right(bytes);
        device_buffer gpu_answer(bytes);

        check_cuda(cudaMemcpy(gpu_left.data, left.cells.data(), bytes, cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(gpu_right.data, right.cells.data(), bytes, cudaMemcpyHostToDevice));

        cudaEvent_t started;
        cudaEvent_t finished;
        check_cuda(cudaEventCreate(&started));
        check_cuda(cudaEventCreate(&finished));

        const dim3 threads(opts.tile, opts.tile);
        const dim3 blocks((answer.order + opts.tile - 1) / opts.tile,
                          (answer.order + opts.tile - 1) / opts.tile);
        const std::size_t shared_bytes = static_cast<std::size_t>(opts.tile) * opts.tile * 2 * sizeof(double);

        check_cuda(cudaEventRecord(started));
        tiled_product<<<blocks, threads, shared_bytes>>>(gpu_left.data, gpu_right.data, gpu_answer.data, answer.order, opts.tile);
        check_cuda(cudaGetLastError());
        check_cuda(cudaEventRecord(finished));
        check_cuda(cudaEventSynchronize(finished));

        float milliseconds = 0.0f;
        check_cuda(cudaEventElapsedTime(&milliseconds, started, finished));
        check_cuda(cudaMemcpy(answer.cells.data(), gpu_answer.data, bytes, cudaMemcpyDeviceToHost));

        cudaEventDestroy(started);
        cudaEventDestroy(finished);

        lab::write_square_matrix(opts.out_path, answer);
        std::cout << "Task scope: " << answer.order << '\n';
        std::cout << "Block size: " << opts.tile << '\n';
        std::cout << "Lead time: " << milliseconds / 1000.0f << " sec\n";
        std::cout << "Operations: " << lab::task_volume(answer.order) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
