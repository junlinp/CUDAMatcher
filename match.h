#ifndef MATCH_H_
#define MATCH_H_

#include <array>
#include <vector>

using Descriptor = std::array<float, 128>;

bool Match(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);

#endif  // MATCH_H_