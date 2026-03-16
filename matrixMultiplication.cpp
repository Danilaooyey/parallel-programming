#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <fstream>

std::vector<long long> readMatrixFromFile(const std::string& filePath, size_t& size) {
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        throw std::runtime_error("[ERROR] => Fail to open matrix file: " + filePath);
    }

    file >> size;
    std::vector<long long> matrix(size * size);

    for (size_t i = 0; i < size * size; i++) {
        file >> matrix[i];
    }
    file.close();
    
    return matrix;
}


std::vector<long long> matrixMultiplication(const std::vector<long long>& matrix1, const std::vector<long long>& matrix2, size_t& size) {
    std::vector<long long> matrinxResult(size * size, 0);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            for (size_t k = 0; k < size; k++) {
                matrinxResult[i*size+j] += matrix1[i*size+k] * matrix2[k*size+j];
            }
        }
    }
    return matrinxResult;
}


void writeMatrixToFile(const std::string& filePath, const std::vector<long long>& matrixResult, size_t size, double time) {
    std::ofstream file(filePath);
    
    if (!file.is_open()) {
        throw std::runtime_error("[ERROR] => Fail to open result file: " + filePath);
    }

    file << size << "\n";

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            file << matrixResult[i * size + j];

            if (j != size - 1) {
                file << " ";
            }
        }
        file << "\n";
    }
    file << "\n";
    file << "Volume: " << size << "\n";
    file << "Time: " << std::to_string(time) << "\n";

    file.close();
}


int main() {
    size_t size1, size2;

    std::vector<long long>matrix1 = readMatrixFromFile("matrix1.txt", size1);
    std::vector<long long>matrix2 = readMatrixFromFile("matrix2.txt", size2);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<long long>result_matrix = matrixMultiplication(matrix1, matrix2, size1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;

    writeMatrixToFile("matrixResult.txt", result_matrix, size1, time.count());
    std::cout << "[RUN] => Matrix multiplication completed. Result written to matrixResult.txt" << std::endl;
}