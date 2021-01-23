
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include "match.h"

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

    
    //std::random_shuffle(shuffle.begin() shuffle.end());

    for (int i = 0; i < descriptor_num; i++) {
        rhs[i] = lhs[shuffle[i]];
    }

    std::vector<std::pair<int, int>> match_result;

    Match(lhs, rhs, match_result);

    for (std::pair<int, int> p : match_result) {
        if (p.second != shuffle[p.first]) {
            std::cout << "error" << std::endl;
        }
    }
}
int main(int argc, char** argv) {
    MATCH();
    return 0;
}