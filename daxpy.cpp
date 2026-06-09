/**
 * Run (4 ranks, 2 OMP threads each):
 *   OMP_NUM_THREADS=2 mpirun -n 4 ./papi_mpi_bench
 *
 * Override N and NREP at compile time:
 *   mpicxx ... -DN=33554432 -DNREP=20 ...
 */

#include <mpi.h>
#include <omp.h>
#include <papi.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

// ── Tuneable knobs (override via -D at compile time) ─────────────────────────
#ifndef N
#  define N (1LL << 24)   // doubles per rank  = 16 777 216  (~128 MB)
#endif
#ifndef NREP
#  define NREP 10         // kernel repetitions
#endif

static constexpr double ALPHA = 2.718281828459045;   // arbitrary, non-trivial
// ─────────────────────────────────────────────────────────────────────────────

static void papi_check(int rc, const char* where) {
    if (rc != PAPI_OK) {
        fprintf(stderr, "[PAPI] %s: %s\n", where, PAPI_strerror(rc));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

int main(int argc, char** argv) {

    // 1. MPI init – request thread support for OpenMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fputs("MPI does not support MPI_THREAD_FUNNELED\n", stderr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    int nthreads = omp_get_max_threads();

    // 2. PAPI library init (once per process)
    // NOTE: PAPI_library_init returns PAPI_VER_CURRENT on success, NOT PAPI_OK.
    // Any other return value means a version mismatch or init failure.
    {
        int papi_ver = PAPI_library_init(PAPI_VER_CURRENT);
        if (papi_ver != PAPI_VER_CURRENT) {
            if (papi_ver > 0)
                fprintf(stderr, "[PAPI] version mismatch: header=%d runtime=%d\n",
                        PAPI_VER_CURRENT, papi_ver);
            else
                fprintf(stderr, "[PAPI] PAPI_library_init failed: %s\n",
                        PAPI_strerror(papi_ver));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    // Register OpenMP thread IDs so PAPI can track per-thread events
    papi_check(
        PAPI_thread_init((unsigned long (*)(void)) omp_get_thread_num),
        "PAPI_thread_init"
    );

    // 3. Print configuration (rank 0 only)
    if (rank == 0) {
        const long long total_flops = 2LL * N * NREP;
        const long long total_bytes = 3LL * 8  * N * NREP;
        printf("====================================================\n");
        printf("  PAPI MPI+OpenMP DAXPY benchmark\n");
        printf("  Ranks              : %d\n",   nranks);
        printf("  OMP threads/rank   : %d\n",   nthreads);
        printf("  N per rank         : %lld  (%.1f MB of doubles)\n",
               (long long)N, N * 8.0 / 1e6);
        printf("  Repetitions        : %d\n",   NREP);
        printf("  Expected FLOPs/rank: %lld\n", total_flops);
        printf("  Expected Bytes/rank: %lld\n", total_bytes);
        printf("====================================================\n");
        fflush(stdout);
    }

    // 4. Allocate and initialise vectors
    std::vector<double> x(N, 1.0);
    std::vector<double> y(N, 0.5);

    // 5. Warm-up pass (not measured)
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; i++)
        y[i] = ALPHA * x[i] + y[i];

    // Reset y so measured passes are deterministic
    std::fill(y.begin(), y.end(), 0.5);

    // 6. PAPI HL region + timed kernel
    // PAPI_hl_region_begin/end must be called by EVERY thread to get per-thread
    // output. We open one persistent parallel region that spans all reps so the
    // team is not torn down and re-created on every iteration.

    MPI_Barrier(MPI_COMM_WORLD);

    const double t0 = MPI_Wtime();

    #pragma omp parallel
    {
        papi_check(PAPI_hl_region_begin("daxpy"), "PAPI_hl_region_begin");

        for (int rep = 0; rep < NREP; rep++) {
            #pragma omp for schedule(static)
            for (long long i = 0; i < N; i++) {
                y[i] = ALPHA * x[i] + y[i];
            }
            // implicit barrier at end of "omp for" keeps reps ordered
        }

        papi_check(PAPI_hl_region_end("daxpy"), "PAPI_hl_region_end");
    }

    const double elapsed = MPI_Wtime() - t0;

    MPI_Barrier(MPI_COMM_WORLD);

    // 7. Prevent dead-code elimination: global checksum
    double local_sum = 0.0;
    for (long long i = 0; i < N; i++) local_sum += y[i];

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    // 8. Gather timings and report
    std::vector<double> all_t(nranks, 0.0);
    MPI_Gather(&elapsed, 1, MPI_DOUBLE,
               all_t.data(), 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n--- Timing results ---\n");
        for (int r = 0; r < nranks; r++) {
            double gflops = (2.0 * N * NREP) / all_t[r] / 1e9;
            double gbps   = (3.0 * 8 * N * NREP) / all_t[r] / 1e9;
            printf("  rank %d: %.3f s  |  %.2f GFlop/s  |  %.2f GB/s\n",
                   r, all_t[r], gflops, gbps);
        }
        printf("  global checksum : %.10e\n", global_sum);
        printf("====================================================\n");
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}