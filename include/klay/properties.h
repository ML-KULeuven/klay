// Copyright (c) 2026 Ibrahim El Kaddouri
// Licensed under apachev2

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "klay/util.h"
#include "klay/circuit.h"

struct SDNNFViolation {
    std::string property;  // e.g. "decomposability"
    int ix;
    std::size_t layer;
    std::size_t node_hash;
    std::string detail;  // e.g. "variable x is in both children"
};

struct SDNNFResult {
    bool is_nnf           = true;  // always true
    bool is_deterministic = true;  // always true if not manually constructed
    bool is_decomposable  = true;
    bool is_smooth        = true;

    std::size_t n_and   = 0;
    std::size_t n_or    = 0;
    std::size_t n_vars_found  = 0;

    std::vector<SDNNFViolation> violations;

    bool is_dnnf()  const { return is_nnf && is_decomposable; }
    // bool is_ddnnf() const { return is_dnnf() && is_deterministic; }
    // bool is_sddnnf() const { return is_ddnnf() && is_smooth; }
    bool is_sdnnf() const { return is_dnnf() && is_smooth; }
};

std::string sdnnf_summary(const SDNNFResult& r);

class IPropertyChecker {
public:
    virtual ~IPropertyChecker() = default;

    virtual void on_node(const Node* node,
                         const SupportMap& scope_of,
                         std::size_t max_violations,
                         SDNNFResult& result) = 0;

    virtual void finalize(std::size_t max_violations,
                          SDNNFResult& result) {}
};

std::unique_ptr<IPropertyChecker> make_decomposability_checker();
// std::unique_ptr<IPropertyChecker> make_determinism_checker();
std::unique_ptr<IPropertyChecker> make_smoothness_checker();


SDNNFResult run_checks(const Circuit& circuit,
                       std::size_t max_violations,
                       std::vector<std::unique_ptr<IPropertyChecker>> checkers);

SDNNFResult check_sdnnf(const Circuit& circuit,
                        std::size_t max_violations = 50);

SDNNFResult check_decomposability(const Circuit& circuit,
                                  std::size_t max_violations = 50);

SDNNFResult check_smooth(const Circuit& circuit,
                         std::size_t max_violations = 50);
