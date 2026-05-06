#pragma once

#include <string>
#include <vector>

#include "klay/circuit.h"

namespace klay {

struct SDNNFViolation {
    std::string property;  // e.g. "decomposability"
    int ix;
    std::size_t layer;
    std::size_t node_hash;
    std::string detail;  // e.g. "variable x is in both children"
};

struct SDNNFResult {
    bool is_nnf           = true;  // always true
    bool is_decomposable  = true;
    bool is_smooth        = true;

    std::size_t n_and   = 0;
    std::size_t n_or    = 0;
    std::size_t n_vars_found  = 0;

    std::vector<SDNNFViolation> violations;

    bool is_dnnf()  const { return is_nnf && is_decomposable; }
    bool is_sdnnf() const { return is_dnnf() && is_smooth; }
};

std::string sdnnf_summary(const SDNNFResult& r);

SDNNFResult check_sdnnf(const Circuit& circuit,
                        std::size_t max_violations = 50);

SDNNFResult check_decomposability(const Circuit& circuit,
                                  std::size_t max_violations = 50);

SDNNFResult check_smooth(const Circuit& circuit,
                         std::size_t max_violations = 50);

}  // namespace klay
