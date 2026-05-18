#include "matrix_io.hpp"

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
    matrix.cells.assign(total, 0.0);

    std::size_t index = 0;
    while (index < total && input >> matrix.cells[index]) {
        ++index;
    }

    if (index != total) {
        throw std::runtime_error("not enough matrix values in file: " + path);
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

    int row = 0;
    while (row < matrix.order) {
        int column = 0;
        while (column < matrix.order) {
            if (column != 0) {
                output << ' ';
            }
            output << matrix.cells[static_cast<std::size_t>(row) * matrix.order + column];
            ++column;
        }
        output << '\n';
        ++row;
    }
}

std::vector<double> multiply_rows(const std::vector<double>& left_rows,
                                  const std::vector<double>& right,
                                  int order,
                                  int row_count) {
    std::vector<double> result(static_cast<std::size_t>(row_count) * order, 0.0);

    int row = 0;
    while (row < row_count) {
        int step = 0;
        while (step < order) {
            const double value = left_rows[static_cast<std::size_t>(row) * order + step];
            const auto right_offset = static_cast<std::size_t>(step) * order;
            const auto result_offset = static_cast<std::size_t>(row) * order;
            int column = 0;
            while (column < order) {
                result[result_offset + column] += value * right[right_offset + column];
                ++column;
            }
            ++step;
        }
        ++row;
    }

    return result;
}

std::size_t task_volume(int order) {
    const auto n = static_cast<std::size_t>(order);
    return n * n * n;
}

}
