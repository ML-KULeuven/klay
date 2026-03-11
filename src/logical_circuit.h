#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>

#include <sstream>
#include <string>
#include <vector>

#include "stratified_dag.h"
#include "literal.h"

namespace nb = nanobind;
using namespace nb::literals;

typedef std::vector<nb::ndarray<nb::numpy, long int, nb::shape<-1>>> Arrays;


class NodePtr {
public:
    NodePtr(Node* ptr) : ptr(ptr) {}

    Node* get() const { return ptr; }

    std::string to_string() const {
        std::stringstream ss;
        ss << "NodePtr(" << this->as_int() << ")";
        return ss.str();
    }

    bool operator==(NodePtr other) const { return this->ptr == other.ptr; }

    std::uintptr_t as_int() const { return reinterpret_cast<std::uintptr_t>(ptr); }

private:
    Node* ptr;
};


/**
 * An AND/OR logical circuit built on top of StratifiedDag.
 *
 * Adds Boolean semantics: dynamic layer assignment via resolve_layer,
 * constant propagation, and SDD/D4 file parsers.
 */
class LogicalCircuit : public StratifiedDag {
public:
    // Gate types — inherent semantic meaning
    static constexpr int Sum = 0;
    static constexpr int Product = 1;

    /**
     * Constant value that acts as the neutral element for the given gate type.
     *   Sum:     neutral = 0 (false)  — x OR false = x
     *   Product: neutral = 1 (true)   — x AND true = x
     */
    static int neutral_value(int gate_type) {
        return (gate_type == Sum) ? 0 : 1;
    }

    /**
     * Constant value that acts as the annihilator for the given gate type.
     *   Sum:     annihilator = 1 (true)  — x OR true = true
     *   Product: annihilator = 0 (false) — x AND false = false
     */
    static int annihilator_value(int gate_type) {
        return (gate_type == Sum) ? 1 : 0;
    }

    void set_root(NodePtr root) {
        StratifiedDag::set_root(root.get());
    }

    using StratifiedDag::remove_unused_nodes;

    /**
     * Like `add_node_level`, but first applies constant propagation:
     *   - Gate with an annihilator child  -> annihilator constant
     *   - Gate with neutral children      -> those children are dropped
     *   - Single remaining child          -> the child itself (no new node)
     *   - No remaining children           -> neutral constant
     *
     * For Gate nodes, reads gate_type from the node and enforces layer
     * placement via the policy before insertion.
     */
    Node* add_node_level_compressed(Node* node);

    std::tuple<Arrays, Arrays, std::vector<int>> get_indices();

    NodePtr true_node();
    NodePtr false_node();
    NodePtr literal_node(int lit);
    NodePtr and_node(std::vector<NodePtr> children);
    NodePtr or_node(std::vector<NodePtr> children);

    NodePtr add_sdd_from_file(const std::string& filename,
                              std::vector<int>& true_lits,
                              std::vector<int>& false_lits);

    NodePtr add_d4_from_file(const std::string& filename,
                             std::vector<int>& true_lits,
                             std::vector<int>& false_lits);

};

// Python-facing alias; preserved for backward compatibility
using Circuit = LogicalCircuit;

void to_dot_file(LogicalCircuit& circuit, const std::string& filename);
