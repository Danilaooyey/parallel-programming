#include "matrix_io.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct run_options {
    std::string left_path = "matrix1.txt";
    std::string right_path = "matrix2.txt";
    std::string out_path = "matrixResult.txt";
};

run_options parse_options(int argc, char** argv) {
    run_options options;
    if (argc > 1) {
        options.left_path = argv[1];
    }
    if (argc > 2) {
        options.right_path = argv[2];
    }
    if (argc > 3) {
        options.out_path = argv[3];
    }
    return options;
}

std::vector<int> split_rows(int order, int workers) {
    std::vector<int> rows(workers, order / workers);
    int extra = order % workers;
    int rank = 0;
    while (rank < extra) {
        ++rows[rank];
        ++rank;
    }
    return rows;
}

std::vector<int> make_counts(const std::vector<int>& rows, int order) {
    std::vector<int> counts(rows.size(), 0);
    std::transform(rows.begin(), rows.end(), counts.begin(), [order](int value) {
        return value * order;
    });
    return counts;
}

std::vector<int> make_offsets(const std::vector<int>& counts) {
    std::vector<int> offsets(counts.size(), 0);
    int index = 1;
    while (index < static_cast<int>(counts.size())) {
        offsets[index] = offsets[index - 1] + counts[index - 1];
        ++index;
    }
    return offsets;
}

}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    const run_options options = parse_options(argc, argv);
    lab::square_matrix left;
    lab::square_matrix right;
    int order = 0;
    int status = 0;

    if (rank == 0) {
        try {
            left = lab::read_square_matrix(options.left_path);
            right = lab::read_square_matrix(options.right_path);
            if (left.order != right.order) {
                throw std::runtime_error("matrix orders are different");
            }
            order = left.order;
        } catch (const std::exception& error) {
            std::cerr << "input error: " << error.what() << '\n';
            status = 1;
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0) {
        MPI_Finalize();
        return status;
    }

    MPI_Bcast(&order, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        right.order = order;
        right.cells.assign(static_cast<std::size_t>(order) * order, 0.0);
    }
    MPI_Bcast(right.cells.data(), order * order, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const std::vector<int> rows = split_rows(order, world);
    const std::vector<int> counts = make_counts(rows, order);
    const std::vector<int> offsets = make_offsets(counts);

    std::vector<double> local_left(static_cast<std::size_t>(counts[rank]), 0.0);
    MPI_Scatterv(rank == 0 ? left.cells.data() : nullptr,
                 counts.data(),
                 offsets.data(),
                 MPI_DOUBLE,
                 local_left.data(),
                 counts[rank],
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double started_at = MPI_Wtime();
    std::vector<double> local_result = lab::multiply_rows(local_left, right.cells, order, rows[rank]);
    const double finished_at = MPI_Wtime();

    double elapsed = finished_at - started_at;
    double slowest = 0.0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    lab::square_matrix answer;
    if (rank == 0) {
        answer.order = order;
        answer.cells.assign(static_cast<std::size_t>(order) * order, 0.0);
    }

    MPI_Gatherv(local_result.data(),
                counts[rank],
                MPI_DOUBLE,
                rank == 0 ? answer.cells.data() : nullptr,
                counts.data(),
                offsets.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        try {
            lab::write_square_matrix(options.out_path, answer);
            std::cout << "Task scope: " << order << '\n';
            std::cout << "Processes: " << world << '\n';
            std::cout << "Lead time: " << slowest << " sec\n";
            std::cout << "Operations: " << lab::task_volume(order) << '\n';
        } catch (const std::exception& error) {
            std::cerr << "output error: " << error.what() << '\n';
            status = 2;
        }
    }

    MPI_Finalize();
    return status;
}
