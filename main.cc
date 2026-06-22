
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <string>
#include "match.h"

struct BenchmarkResult {
    bool success;
    long long elapsed_ms;
    int mismatch_count;
};

BenchmarkResult RunBenchmark(const std::string& name,
                             bool (*matcher)(const std::vector<Descriptor>&,
                                             const std::vector<Descriptor>&,
                                             std::vector<std::pair<int, int>>&),
                             const std::vector<Descriptor>& lhs,
                             const std::vector<Descriptor>& rhs,
                             const std::vector<int>& expected_match) {
    std::vector<std::pair<int, int>> match_result;
    auto start = std::chrono::high_resolution_clock::now();
    bool success = matcher(lhs, rhs, match_result);
    auto end = std::chrono::high_resolution_clock::now();
    long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    int mismatch_count = 0;
    if (success) {
        for (std::pair<int, int> p : match_result) {
            if (p.second != expected_match[p.first]) {
                mismatch_count++;
            }
        }
    } else {
        mismatch_count = static_cast<int>(expected_match.size());
    }

    std::cout << name << " time elapsed : " << elapsed_ms << " ms" << std::endl;
    std::cout << name << " mismatch count : " << mismatch_count << std::endl;
    if (!success) {
        std::cout << name << " match failed" << std::endl;
    }

    BenchmarkResult result;
    result.success = success;
    result.elapsed_ms = elapsed_ms;
    result.mismatch_count = mismatch_count;
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

    BenchmarkResult v1 = RunBenchmark("v1", MatchV1, lhs, rhs, expected_match);
    BenchmarkResult v2 = RunBenchmark("v2", MatchV2, lhs, rhs, expected_match);
    BenchmarkResult v3 = RunBenchmark("v3", MatchV3, lhs, rhs, expected_match);
    BenchmarkResult v4 = RunBenchmark("v4", MatchV4, lhs, rhs, expected_match);
    BenchmarkResult v5 = RunBenchmark("v5", MatchV5, lhs, rhs, expected_match);

    if (v1.success && v2.success && v1.mismatch_count == 0 && v2.mismatch_count == 0 && v2.elapsed_ms > 0) {
        double speedup = static_cast<double>(v1.elapsed_ms) / static_cast<double>(v2.elapsed_ms);
        std::cout << "v2 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v1.success && v3.success && v1.mismatch_count == 0 && v3.mismatch_count == 0 && v3.elapsed_ms > 0) {
        double speedup = static_cast<double>(v1.elapsed_ms) / static_cast<double>(v3.elapsed_ms);
        std::cout << "v3 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v2.success && v3.success && v2.mismatch_count == 0 && v3.mismatch_count == 0 && v3.elapsed_ms > 0) {
        double speedup = static_cast<double>(v2.elapsed_ms) / static_cast<double>(v3.elapsed_ms);
        std::cout << "v3 speedup over v2 : " << speedup << "x" << std::endl;
    }
    if (v1.success && v4.success && v1.mismatch_count == 0 && v4.mismatch_count == 0 && v4.elapsed_ms > 0) {
        double speedup = static_cast<double>(v1.elapsed_ms) / static_cast<double>(v4.elapsed_ms);
        std::cout << "v4 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v3.success && v4.success && v3.mismatch_count == 0 && v4.mismatch_count == 0 && v4.elapsed_ms > 0) {
        double speedup = static_cast<double>(v3.elapsed_ms) / static_cast<double>(v4.elapsed_ms);
        std::cout << "v4 speedup over v3 : " << speedup << "x" << std::endl;
    }
    if (v1.success && v5.success && v1.mismatch_count == 0 && v5.mismatch_count == 0 && v5.elapsed_ms > 0) {
        double speedup = static_cast<double>(v1.elapsed_ms) / static_cast<double>(v5.elapsed_ms);
        std::cout << "v5 speedup over v1 : " << speedup << "x" << std::endl;
    }
    if (v4.success && v5.success && v4.mismatch_count == 0 && v5.mismatch_count == 0 && v5.elapsed_ms > 0) {
        double speedup = static_cast<double>(v4.elapsed_ms) / static_cast<double>(v5.elapsed_ms);
        std::cout << "v5 speedup over v4 : " << speedup << "x" << std::endl;
    }
    if (v3.success && v5.success && v3.mismatch_count == 0 && v5.mismatch_count == 0 && v5.elapsed_ms > 0) {
        double speedup = static_cast<double>(v3.elapsed_ms) / static_cast<double>(v5.elapsed_ms);
        std::cout << "v5 speedup over v3 : " << speedup << "x" << std::endl;
    }
}
int main(int argc, char** argv) {
    MATCH();
    return 0;
}
