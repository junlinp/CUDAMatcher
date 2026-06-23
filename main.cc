
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
#include "cuda_runtime.h"
#include "match.h"

struct BenchmarkResult {
    bool success;
    double wall_best_ms;
    double wall_avg_ms;
    float device_best_ms;
    float device_avg_ms;
    int mismatch_count;
};

double EstimatedFlops(const std::string& name, size_t descriptor_num) {
    double n = static_cast<double>(descriptor_num);
    double k = 128.0;
    if (name == "v9") {
        return n * n * (2.0 * k + 3.0);
    }
    return n * n * (3.0 * k);
}

double EstimatedMemoryBytes(const std::string& name, size_t descriptor_num) {
    double n = static_cast<double>(descriptor_num);
    double k = 128.0;
    if (name == "v1") {
        double descriptor_scan = (n / 32.0) * (32.0 * k * 4.0 + n * k * 4.0);
        double matrix_write_read = 2.0 * n * n * 4.0;
        return descriptor_scan + matrix_write_read;
    }
    if (name == "v4") {
        return (n / 8.0) * (8.0 * k * 4.0 + n * k * 4.0);
    }
    if (name == "v8") {
        return (n / 32.0) * (32.0 * k * 2.0 + n * k * 2.0);
    }
    if (name == "v9") {
        double descriptor_scan = (n / 16.0) * (16.0 * k * 2.0 + n * k * 2.0);
        double norm_scan = (n / 16.0) * (16.0 * 4.0 + n * 4.0);
        return descriptor_scan + norm_scan;
    }
    return (n / 32.0) * (32.0 * k * 4.0 + n * k * 4.0);
}

void PrintMetrics(const std::string& name, const BenchmarkResult& result, size_t descriptor_num) {
    if (!result.success || result.wall_best_ms <= 0.0) {
        return;
    }
    double seconds = result.wall_best_ms / 1000.0;
    double gflops = EstimatedFlops(name, descriptor_num) / seconds / 1e9;
    double bandwidth = EstimatedMemoryBytes(name, descriptor_num) / seconds / 1e9;
    std::cout << name << " wall best : " << std::fixed << std::setprecision(3) << result.wall_best_ms << " ms" << std::endl;
    std::cout << name << " wall avg : " << std::fixed << std::setprecision(3) << result.wall_avg_ms << " ms" << std::endl;
    std::cout << name << " wall-best estimated compute : " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s" << std::endl;
    std::cout << name << " wall-best estimated bandwidth : " << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
    if (result.device_best_ms > 0.0f) {
        double device_seconds = static_cast<double>(result.device_best_ms) / 1000.0;
        double device_gflops = EstimatedFlops(name, descriptor_num) / device_seconds / 1e9;
        double device_bandwidth = EstimatedMemoryBytes(name, descriptor_num) / device_seconds / 1e9;
        std::cout << name << " device best : " << std::fixed << std::setprecision(3) << result.device_best_ms << " ms" << std::endl;
        std::cout << name << " device avg : " << std::fixed << std::setprecision(3) << result.device_avg_ms << " ms" << std::endl;
        std::cout << name << " device-best estimated compute : " << std::fixed << std::setprecision(2) << device_gflops << " GFLOP/s" << std::endl;
        std::cout << name << " device-best estimated bandwidth : " << std::fixed << std::setprecision(2) << device_bandwidth << " GB/s" << std::endl;
    }
}

void PrintKernelMetrics(const std::string& name, const KernelBenchmarkResult& result, size_t descriptor_num) {
    std::cout << name << " kernel-only mismatch count : " << result.mismatch_count << std::endl;
    if (!result.success || result.best_ms <= 0.0f) {
        std::cout << name << " kernel-only benchmark failed" << std::endl;
        return;
    }
    double seconds = static_cast<double>(result.best_ms) / 1000.0;
    double gflops = EstimatedFlops(name, descriptor_num) / seconds / 1e9;
    double bandwidth = EstimatedMemoryBytes(name, descriptor_num) / seconds / 1e9;
    std::cout << name << " kernel-only best : " << std::fixed << std::setprecision(3) << result.best_ms << " ms" << std::endl;
    std::cout << name << " kernel-only avg : " << std::fixed << std::setprecision(3) << result.avg_ms << " ms" << std::endl;
    std::cout << name << " kernel-only estimated compute : " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s" << std::endl;
    std::cout << name << " kernel-only estimated bandwidth : " << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
}

void RunKernelOnlyBenchmarks(const std::vector<Descriptor>& lhs,
                             const std::vector<Descriptor>& rhs,
                             const std::vector<int>& expected_match,
                             size_t descriptor_num) {
    const int warmup_runs = 5;
    const int measured_runs = 20;
    KernelBenchmarkResult v1;
    KernelBenchmarkResult v2;
    KernelBenchmarkResult v3;
    KernelBenchmarkResult v4;
    KernelBenchmarkResult v5;
    KernelBenchmarkResult v5a;
    KernelBenchmarkResult v5b;
    KernelBenchmarkResult v5c;
    KernelBenchmarkResult v6;
    KernelBenchmarkResult v7;
    KernelBenchmarkResult v8;
    KernelBenchmarkResult v9;

    std::cout << "kernel-only measured runs : " << measured_runs << " + " << warmup_runs << " warmup" << std::endl;
    BenchmarkKernelV1(lhs, rhs, expected_match, warmup_runs, measured_runs, v1);
    PrintKernelMetrics("v1", v1, descriptor_num);
    BenchmarkKernelV2(lhs, rhs, expected_match, warmup_runs, measured_runs, v2);
    PrintKernelMetrics("v2", v2, descriptor_num);
    BenchmarkKernelV3(lhs, rhs, expected_match, warmup_runs, measured_runs, v3);
    PrintKernelMetrics("v3", v3, descriptor_num);
    BenchmarkKernelV4(lhs, rhs, expected_match, warmup_runs, measured_runs, v4);
    PrintKernelMetrics("v4", v4, descriptor_num);
    BenchmarkKernelV5(lhs, rhs, expected_match, warmup_runs, measured_runs, v5);
    PrintKernelMetrics("v5", v5, descriptor_num);
    BenchmarkKernelV5a(lhs, rhs, expected_match, warmup_runs, measured_runs, v5a);
    PrintKernelMetrics("v5a", v5a, descriptor_num);
    BenchmarkKernelV5b(lhs, rhs, expected_match, warmup_runs, measured_runs, v5b);
    PrintKernelMetrics("v5b", v5b, descriptor_num);
    BenchmarkKernelV5c(lhs, rhs, expected_match, warmup_runs, measured_runs, v5c);
    PrintKernelMetrics("v5c", v5c, descriptor_num);
    BenchmarkKernelV6(lhs, rhs, expected_match, warmup_runs, measured_runs, v6);
    PrintKernelMetrics("v6", v6, descriptor_num);
    BenchmarkKernelV7(lhs, rhs, expected_match, warmup_runs, measured_runs, v7);
    PrintKernelMetrics("v7", v7, descriptor_num);
    BenchmarkKernelV8(lhs, rhs, expected_match, warmup_runs, measured_runs, v8);
    PrintKernelMetrics("v8", v8, descriptor_num);
    BenchmarkKernelV9(lhs, rhs, expected_match, warmup_runs, measured_runs, v9);
    PrintKernelMetrics("v9", v9, descriptor_num);
}

BenchmarkResult RunBenchmark(const std::string& name,
                             bool (*matcher)(const std::vector<Descriptor>&,
                                             const std::vector<Descriptor>&,
                                             std::vector<std::pair<int, int>>&),
                             const std::vector<Descriptor>& lhs,
                             const std::vector<Descriptor>& rhs,
                             const std::vector<int>& expected_match,
                             size_t descriptor_num) {
    const int warmup_runs = 1;
    const int measured_runs = 5;
    bool success = true;
    int mismatch_count = 0;
    double wall_sum_ms = 0.0;
    double wall_best_ms = std::numeric_limits<double>::max();
    float device_sum_ms = 0.0f;
    float device_best_ms = std::numeric_limits<float>::max();

    for (int run = 0; run < warmup_runs + measured_runs; run++) {
        std::vector<std::pair<int, int>> match_result;
        cudaEvent_t device_start = nullptr;
        cudaEvent_t device_stop = nullptr;
        float device_elapsed_ms = 0.0f;
        cudaEventCreate(&device_start);
        cudaEventCreate(&device_stop);
        cudaEventRecord(device_start, 0);
        auto start = std::chrono::high_resolution_clock::now();
        bool run_success = matcher(lhs, rhs, match_result);
        auto end = std::chrono::high_resolution_clock::now();
        cudaEventRecord(device_stop, 0);
        cudaEventSynchronize(device_stop);
        cudaEventElapsedTime(&device_elapsed_ms, device_start, device_stop);
        cudaEventDestroy(device_start);
        cudaEventDestroy(device_stop);

        double wall_elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        int run_mismatch_count = 0;
        if (run_success) {
            for (std::pair<int, int> p : match_result) {
                if (p.second != expected_match[p.first]) {
                    run_mismatch_count++;
                }
            }
        } else {
            run_mismatch_count = static_cast<int>(expected_match.size());
        }

        success = success && run_success;
        if (run_mismatch_count > mismatch_count) {
            mismatch_count = run_mismatch_count;
        }

        if (run >= warmup_runs) {
            wall_sum_ms += wall_elapsed_ms;
            device_sum_ms += device_elapsed_ms;
            if (wall_elapsed_ms < wall_best_ms) {
                wall_best_ms = wall_elapsed_ms;
            }
            if (device_elapsed_ms < device_best_ms) {
                device_best_ms = device_elapsed_ms;
            }
        }
    }

    double wall_avg_ms = wall_sum_ms / static_cast<double>(measured_runs);
    float device_avg_ms = device_sum_ms / static_cast<float>(measured_runs);

    std::cout << name << " measured runs : " << measured_runs << " + " << warmup_runs << " warmup" << std::endl;
    std::cout << name << " mismatch count : " << mismatch_count << std::endl;
    if (!success) {
        std::cout << name << " match failed" << std::endl;
    }

    BenchmarkResult result;
    result.success = success;
    result.wall_best_ms = wall_best_ms;
    result.wall_avg_ms = wall_avg_ms;
    result.device_best_ms = device_best_ms;
    result.device_avg_ms = device_avg_ms;
    result.mismatch_count = mismatch_count;
    PrintMetrics(name, result, descriptor_num);
    return result;
}

void MATCH() {
    const size_t descriptor_num = 1024 * 16;

    std::vector<Descriptor> lhs(descriptor_num), rhs(descriptor_num);
    std::vector<int> shuffle(descriptor_num);
    std::default_random_engine engine;
    std::uniform_real_distribution<float> distribute(0, 255);
    for (int i = 0; i < descriptor_num; i++) {
        Descriptor descriptor;
        for (int k  = 0; k < 128; k++) {
            descriptor[k] = distribute(engine);
        }
        shuffle[i] = descriptor_num - 1 - i;
        lhs[i] = descriptor;
    }

    std::random_shuffle(shuffle.begin(), shuffle.end());
    for (int i = 0; i < descriptor_num; i++) {
        rhs[i] = lhs[shuffle[i]];
    }
    std::vector<int> expected_match(descriptor_num);
    for (int i = 0; i < descriptor_num; i++) {
        expected_match[shuffle[i]] = i;
    }

    BenchmarkResult v1 = RunBenchmark("v1", MatchV1, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v2 = RunBenchmark("v2", MatchV2, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v3 = RunBenchmark("v3", MatchV3, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v4 = RunBenchmark("v4", MatchV4, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v5 = RunBenchmark("v5", MatchV5, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v5a = RunBenchmark("v5a", MatchV5a, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v5b = RunBenchmark("v5b", MatchV5b, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v5c = RunBenchmark("v5c", MatchV5c, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v6 = RunBenchmark("v6", MatchV6, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v7 = RunBenchmark("v7", MatchV7, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v8 = RunBenchmark("v8", MatchV8, lhs, rhs, expected_match, descriptor_num);
    BenchmarkResult v9 = RunBenchmark("v9", MatchV9, lhs, rhs, expected_match, descriptor_num);

    if (v1.success && v2.success && v1.mismatch_count == 0 && v2.mismatch_count == 0 && v2.wall_best_ms > 0) {
        double speedup = static_cast<double>(v1.wall_best_ms) / static_cast<double>(v2.wall_best_ms);
        std::cout << "v2 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v1.success && v3.success && v1.mismatch_count == 0 && v3.mismatch_count == 0 && v3.wall_best_ms > 0) {
        double speedup = static_cast<double>(v1.wall_best_ms) / static_cast<double>(v3.wall_best_ms);
        std::cout << "v3 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v2.success && v3.success && v2.mismatch_count == 0 && v3.mismatch_count == 0 && v3.wall_best_ms > 0) {
        double speedup = static_cast<double>(v2.wall_best_ms) / static_cast<double>(v3.wall_best_ms);
        std::cout << "v3 speedup over v2 : " << speedup << "x" << std::endl;
    }
    if (v1.success && v4.success && v1.mismatch_count == 0 && v4.mismatch_count == 0 && v4.wall_best_ms > 0) {
        double speedup = static_cast<double>(v1.wall_best_ms) / static_cast<double>(v4.wall_best_ms);
        std::cout << "v4 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v3.success && v4.success && v3.mismatch_count == 0 && v4.mismatch_count == 0 && v4.wall_best_ms > 0) {
        double speedup = static_cast<double>(v3.wall_best_ms) / static_cast<double>(v4.wall_best_ms);
        std::cout << "v4 speedup over v3 : " << speedup << "x" << std::endl;
    }
    if (v1.success && v5.success && v1.mismatch_count == 0 && v5.mismatch_count == 0 && v5.wall_best_ms > 0) {
        double speedup = static_cast<double>(v1.wall_best_ms) / static_cast<double>(v5.wall_best_ms);
        std::cout << "v5 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v4.success && v5.success && v4.mismatch_count == 0 && v5.mismatch_count == 0 && v5.wall_best_ms > 0) {
        double speedup = static_cast<double>(v4.wall_best_ms) / static_cast<double>(v5.wall_best_ms);
        std::cout << "v5 speedup over v4 : " << speedup << "x" << std::endl;
    }
    if (v3.success && v5.success && v3.mismatch_count == 0 && v5.mismatch_count == 0 && v5.wall_best_ms > 0) {
        double speedup = static_cast<double>(v3.wall_best_ms) / static_cast<double>(v5.wall_best_ms);
        std::cout << "v5 speedup over v3 : " << speedup << "x" << std::endl;
    }
    if (v5.success && v5a.success && v5.mismatch_count == 0 && v5a.mismatch_count == 0 && v5a.wall_best_ms > 0) {
        double speedup = static_cast<double>(v5.wall_best_ms) / static_cast<double>(v5a.wall_best_ms);
        std::cout << "v5a speedup over v5 : " << speedup << "x" << std::endl;
    }
    if (v5a.success && v5b.success && v5a.mismatch_count == 0 && v5b.mismatch_count == 0 && v5b.wall_best_ms > 0) {
        double speedup = static_cast<double>(v5a.wall_best_ms) / static_cast<double>(v5b.wall_best_ms);
        std::cout << "v5b speedup over v5a : " << speedup << "x" << std::endl;
    }
    if (v5b.success && v5c.success && v5b.mismatch_count == 0 && v5c.mismatch_count == 0 && v5c.wall_best_ms > 0) {
        double speedup = static_cast<double>(v5b.wall_best_ms) / static_cast<double>(v5c.wall_best_ms);
        std::cout << "v5c speedup over v5b : " << speedup << "x" << std::endl;
    }
    if (v4.success && v6.success && v4.mismatch_count == 0 && v6.mismatch_count == 0 && v6.wall_best_ms > 0) {
        double speedup = static_cast<double>(v4.wall_best_ms) / static_cast<double>(v6.wall_best_ms);
        std::cout << "v6 speedup over v4 : " << speedup << "x" << std::endl;
    }
    if (v5.success && v6.success && v5.mismatch_count == 0 && v6.mismatch_count == 0 && v6.wall_best_ms > 0) {
        double speedup = static_cast<double>(v5.wall_best_ms) / static_cast<double>(v6.wall_best_ms);
        std::cout << "v6 speedup over v5 : " << speedup << "x" << std::endl;
    }
    if (v6.success && v7.success && v6.mismatch_count == 0 && v7.mismatch_count == 0 && v7.wall_best_ms > 0) {
        double speedup = static_cast<double>(v6.wall_best_ms) / static_cast<double>(v7.wall_best_ms);
        std::cout << "v7 speedup over v6 : " << speedup << "x" << std::endl;
    }
    if (v6.success && v8.success && v6.mismatch_count == 0 && v8.mismatch_count == 0 && v8.wall_best_ms > 0) {
        double speedup = static_cast<double>(v6.wall_best_ms) / static_cast<double>(v8.wall_best_ms);
        std::cout << "v8 speedup over v6 : " << speedup << "x" << std::endl;
    }
    if (v8.success && v9.success && v8.mismatch_count == 0 && v9.mismatch_count == 0 && v9.wall_best_ms > 0) {
        double speedup = static_cast<double>(v8.wall_best_ms) / static_cast<double>(v9.wall_best_ms);
        std::cout << "v9 speedup over v8 : " << speedup << "x" << std::endl;
    }
    if (v6.success && v9.success && v6.mismatch_count == 0 && v9.mismatch_count == 0 && v9.wall_best_ms > 0) {
        double speedup = static_cast<double>(v6.wall_best_ms) / static_cast<double>(v9.wall_best_ms);
        std::cout << "v9 speedup over v6 : " << speedup << "x" << std::endl;
    }

    RunKernelOnlyBenchmarks(lhs, rhs, expected_match, descriptor_num);
}
int main(int argc, char** argv) {
    MATCH();
    return 0;
}
