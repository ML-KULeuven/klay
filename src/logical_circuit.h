#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>

#include <sstream>
#include <string>
#include <vector>

#include "stratified_dag.h"
#include "layer_policy.h"
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
 * An AND/OR logical circuit built on top of StratifiedDag<AndOrPolicy>.
 *
 * Adds Boolean semantics: And/Or/Literal/True/False node types, layer-parity
 * enforcement (And=odd, Or=even), constant propagation, and SDD/D4 file parsers.
 */
class LogicalCircuit : public StratifiedDag<AndOrPolicy> {
public:

    void set_root(NodePtr root) {
        StratifiedDag<AndOrPolicy>::set_root(root.get());
    }

    using StratifiedDag<AndOrPolicy>::remove_unused_nodes;

    /**
     * Like `add_node_level`, but first applies Boolean simplification:
     *   - OR with a True child  -> True
     *   - OR with False children -> those children are dropped
     *   - AND with a False child -> False
     *   - AND with True children -> those children are dropped
     *   - Single remaining child  -> the child itself (no new node)
     *   - No remaining children   -> neutral element (True for AND, False for OR)
     *
     * Also enforces layer parity (And=odd, Or=even) before insertion.
     */
    Node* add_node_level_compressed(Node* node);

    std::pair<Arrays, Arrays> get_indices();

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
