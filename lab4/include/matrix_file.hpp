#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace lab {

struct square_matrix {
    int order = 0;
    std::vector<double> cells;
};

square_matrix read_square_matrix(const std::string& path);
void write_square_matrix(const std::string& path, const square_matrix& matrix);
std::size_t task_volume(int order);

}
