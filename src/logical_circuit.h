#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "stratified_dag.h"


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

    static int neutral_value(int gate_type)     { return (gate_type == Sum) ? 0 : 1; }
    static int annihilator_value(int gate_type) { return (gate_type == Sum) ? 1 : 0; }

    using StratifiedDag::set_root;
    using StratifiedDag::remove_unused_nodes;

    /**
     * Like `add_node_level`, but first applies constant propagation:
     *   - Gate with an annihilator child  -> annihilator constant
     *   - Gate with neutral children      -> those children are dropped
     *   - Single remaining child          -> the child itself (no new node)
     *   - No remaining children           -> neutral constant
     */
    Node* add_node_level_compressed(Node* node);

    /**
     * Returns the CSR indices and pointers for each layer, plus the gate type per layer.
     * indices[i] contains child indices for layer i+1.
     * csr[i] contains the row pointers (size = layer_size + 1) for layer i+1.
     */
    std::tuple<std::vector<std::vector<long>>, std::vector<std::vector<long>>, std::vector<int>> get_indices();

    Node* true_node();
    Node* false_node();
    Node* literal_node(int lit);
    Node* and_node(std::vector<Node*> children);
    Node* or_node(std::vector<Node*> children);

    Node* add_sdd_from_file(const std::string& filename,
                            std::vector<int>& true_lits,
                            std::vector<int>& false_lits);

    Node* add_d4_from_file(const std::string& filename,
                           std::vector<int>& true_lits,
                           std::vector<int>& false_lits);
};

void to_dot_file(LogicalCircuit& circuit, const std::string& filename);
