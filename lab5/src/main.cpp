#include "matrix_io.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct paths {
    std::string left = "matrix1.txt";
    std::string right = "matrix2.txt";
    std::string result = "matrixResult.txt";
};

paths parse_paths(int argc, char** argv) {
    paths value;
    if (argc > 1) {
        value.left = argv[1];
    }
    if (argc > 2) {
        value.right = argv[2];
    }
    if (argc > 3) {
        value.result = argv[3];
    }
    return value;
}

std::vector<int> row_plan(int n, int ranks) {
    std::vector<int> rows(ranks, n / ranks);
    for (int i = 0; i < n % ranks; ++i) {
        ++rows[i];
    }
    return rows;
}

std::vector<int> element_counts(const std::vector<int>& rows, int n) {
    std::vector<int> counts;
    counts.reserve(rows.size());
    for (int item : rows) {
        counts.push_back(item * n);
    }
    return counts;
}

std::vector<int> displacements(const std::vector<int>& counts) {
    std::vector<int> result(counts.size(), 0);
    for (int i = 1; i < static_cast<int>(counts.size()); ++i) {
        result[i] = result[i - 1] + counts[i - 1];
    }
    return result;
}

}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    const paths file = parse_paths(argc, argv);
    int n = 0;
    int failed = 0;
    lab::square_matrix a;
    lab::square_matrix b;

    if (rank == 0) {
        try {
            a = lab::read_square_matrix(file.left);
            b = lab::read_square_matrix(file.right);
            if (a.order != b.order) {
                throw std::runtime_error("matrix orders differ");
            }
            n = a.order;
        } catch (const std::exception& error) {
            std::cerr << "read failed: " << error.what() << '\n';
            failed = 1;
        }
    }

    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (failed != 0) {
        MPI_Finalize();
        return failed;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        b.order = n;
        b.cells.assign(static_cast<std::size_t>(n) * n, 0.0);
    }
    MPI_Bcast(b.cells.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const std::vector<int> rows = row_plan(n, ranks);
    const std::vector<int> counts = element_counts(rows, n);
    const std::vector<int> shifts = displacements(counts);

    std::vector<double> local_a(static_cast<std::size_t>(counts[rank]), 0.0);
    MPI_Scatterv(rank == 0 ? a.cells.data() : nullptr,
                 counts.data(),
                 shifts.data(),
                 MPI_DOUBLE,
                 local_a.data(),
                 counts[rank],
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    std::vector<double> local_c = lab::multiply_rows(local_a, b.cells, n, rows[rank]);
    const double t1 = MPI_Wtime();
    const double local_seconds = t1 - t0;
    double seconds = 0.0;
    MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    lab::square_matrix c;
    if (rank == 0) {
        c.order = n;
        c.cells.assign(static_cast<std::size_t>(n) * n, 0.0);
    }

    MPI_Gatherv(local_c.data(),
                counts[rank],
                MPI_DOUBLE,
                rank == 0 ? c.cells.data() : nullptr,
                counts.data(),
                shifts.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (rank == 0) {
        try {
            lab::write_square_matrix(file.result, c);
            std::cout << "Task scope: " << n << '\n';
            std::cout << "MPI ranks: " << ranks << '\n';
            std::cout << "Lead time: " << seconds << " sec\n";
            std::cout << "Operations: " << lab::task_volume(n) << '\n';
        } catch (const std::exception& error) {
            std::cerr << "write failed: " << error.what() << '\n';
            failed = 2;
        }
    }

    MPI_Finalize();
    return failed;
}
