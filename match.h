#ifndef MATCH_H_
#define MATCH_H_

#include <array>
#include <vector>

using Descriptor = std::array<float, 128>;

bool MatchV1(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool MatchV2(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);
bool Match(const std::vector< Descriptor>& lhs,const std::vector<Descriptor>& rhs, std::vector<std::pair<int, int>>& match_result);

#endif  // MATCH_H_
