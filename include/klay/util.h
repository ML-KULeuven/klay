#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "klay/node.h"

namespace klay {

using Word       = std::uint64_t;
using Support    = std::vector<Word>;
using SupportMap = std::unordered_map<const Node*, Support>;

inline std::size_t
n_words(std::size_t n_vars) {
    return (n_vars + 63) / 64;
}

inline void
support_var(Support& s,
            std::size_t v) {
    std::size_t bit = v - 1;
    std::size_t word_idx = bit / 64;
    std::size_t bit_idx  = bit & 63;  // bit % 64
    s[word_idx] |= (Word(1) << bit_idx);
}

inline void
support_union(Support& dst,
              const Support& src) {
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] |= src[i];
}

inline bool
support_intersect(const Support& a,
                  const Support& b) {
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] & b[i]) return true;
    return false;
}

inline bool
support_equal(const Support& a,
              const Support& b) {
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

inline std::string
support_to_string(const Support& s,
                  std::size_t n_vars) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (std::size_t v = 1; v <= n_vars; ++v) {
        std::size_t bit = v - 1;
        if (s[bit >> 6] & (Word(1) << (bit & 63))) {
            if (!first) oss << ", ";
            oss << v;
            first = false;
        }
    }
    oss << "}";
    return oss.str();
}

inline std::string
support_sym_diff_string(const Support& a,
                        const Support& b,
                        std::size_t n_vars) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (std::size_t v = 1; v <= n_vars; ++v) {
        std::size_t bit = v - 1;
        Word wa = a[bit >> 6] & (Word(1) << (bit & 63));
        Word wb = b[bit >> 6] & (Word(1) << (bit & 63));
        if ((wa != 0) != (wb != 0)) {  // XOR
            if (!first) oss << ", ";
            oss << v;
            first = false;
        }
    }
    oss << "}";
    return oss.str();
}

}  // namespace klay
