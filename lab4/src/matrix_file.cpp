#include "matrix_file.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace lab {

square_matrix read_square_matrix(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + path);
    }

    square_matrix matrix;
    input >> matrix.order;
    if (!input || matrix.order <= 0) {
        throw std::runtime_error("bad matrix size in file: " + path);
    }

    const auto total = static_cast<std::size_t>(matrix.order) * matrix.order;
    matrix.cells.resize(total);
    for (double& value : matrix.cells) {
        if (!(input >> value)) {
            throw std::runtime_error("not enough values in file: " + path);
        }
    }

    return matrix;
}

void write_square_matrix(const std::string& path, const square_matrix& matrix) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot create output file: " + path);
    }

    output << matrix.order << '\n';
    output << std::fixed << std::setprecision(8);
    for (int row = 0; row < matrix.order; ++row) {
        for (int column = 0; column < matrix.order; ++column) {
            if (column > 0) {
                output << ' ';
            }
            output << matrix.cells[static_cast<std::size_t>(row) * matrix.order + column];
        }
        output << '\n';
    }
}

std::size_t task_volume(int order) {
    const auto n = static_cast<std::size_t>(order);
    return n * n * n;
}

}
