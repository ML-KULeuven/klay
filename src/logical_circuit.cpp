#include "logical_circuit.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>


// ---------------------------------------------------------------------------
// LogicalCircuit
// ---------------------------------------------------------------------------

Node* LogicalCircuit::add_node_level_compressed(Node* node) {
    // Non-gate nodes (constants, leaves) have no children to propagate through.
    if (node->type != NodeType::Gate)
        return add_node_level(node);

    // Pin the node to the correct layer for its gate type before inspecting children.
    node->layer = resolve_layer(node->gate_type, node->layer);

    int neutral     = neutral_value(node->gate_type);
    int annihilator = annihilator_value(node->gate_type);

    std::list<Node*> new_children;
    for (auto& child : node->children) {
        if (child->is_constant(neutral)) {
            continue;                            // neutral element: drop (e.g. AND drops True children)
        } else if (child->is_constant(annihilator)) {
            delete node;                         // annihilator absorbs: whole gate collapses
            return add_node_level(Node::createConstant(annihilator));
        } else {
            new_children.push_back(child);
        }
    }

    if (new_children.empty()) {
        // All children were neutral — gate reduces to neutral constant.
        delete node;
        return add_node_level(Node::createConstant(neutral));
    }
    if (new_children.size() == 1) {
        // Single child remaining — gate is redundant, bypass it.
        Node* child = new_children.front();
        delete node;
        return child;
    }
    if (new_children.size() != node->children.size()) {
        // Some children were dropped; the hash was computed from the original
        // child set, so we must recreate the node to get the correct hash.
        int gt = node->gate_type;
        delete node;
        node = Node::createGate(gt);
        for (auto child : new_children)
            node->add_child(child);
        node->layer = resolve_layer(node->gate_type, node->layer);
    }

    return add_node_level(node);
}


std::tuple<std::vector<std::vector<long>>, std::vector<std::vector<long>>, std::vector<int>> LogicalCircuit::get_indices() {
    remove_unused_nodes();
    add_root_layer();

    std::vector<std::vector<long>> indices_vecs; // child index arrays, one per layer transition
    std::vector<std::vector<long>> csr_vecs;     // CSR row-pointer arrays, one per layer transition

    for (std::size_t i = 1; i < nb_layers(); ++i) {
        // Count children per node, indexed by node->ix within this layer.
        std::vector<long> child_counts(layers[i].size(), 0);
        std::size_t layer_size = 0;             // total edges in this layer
        for (const auto* node : layers[i]) {
            layer_size += node->children.size();
            child_counts[node->ix] = node->children.size();
        }

        // Build CSR row pointers: csr[k] = start index of node k's children in `indices`.
        std::vector<long> csr(layers[i].size() + 1);
        csr[0] = 0;
        for (std::size_t j = 1; j < csr.size(); ++j)
            csr[j] = csr[j-1] + child_counts[j-1];

        // Fill flat child-index array using the CSR offsets.
        std::vector<long> indices(layer_size);
        for (const auto* node : layers[i]) {
            std::size_t offset = 0;
            for (Node* child : node->children) {
                assert(child->layer == i - 1);  // children must be in the adjacent lower layer
                indices[csr[node->ix] + offset++] = child->ix;
            }
        }

        indices_vecs.push_back(std::move(indices));
        csr_vecs.push_back(std::move(csr));
    }

    // Collect gate type per layer (skip layer 0, the leaf/constant layer).
    std::vector<int> comp_layer_types;
    for (std::size_t i = 1; i < nb_layers(); ++i)
        comp_layer_types.push_back(layer_gate_types[i]);

    return {std::move(indices_vecs), std::move(csr_vecs), comp_layer_types};
}


Node* LogicalCircuit::true_node() {
    return add_node_level(Node::createConstant(1));
}

Node* LogicalCircuit::false_node() {
    return add_node_level(Node::createConstant(0));
}

Node* LogicalCircuit::literal_node(int lit) {
    if (lit == 0)
        throw std::domain_error(
            "literal_node(lit) does not allow lit == 0, "
            "because negation -0 does not make sense.");
    Lit l = Lit::fromInt(lit);
    // Positive and negative literals share a layer; mix_hash distinguishes them.
    return add_node_level(Node::createLeaf(l.internal_val(), mix_hash(l.internal_val())));
}

Node* LogicalCircuit::and_node(std::vector<Node*> children) {
    Node* node = Node::createGate(Product);
    for (auto child : children)
        node->add_child(child);
    return add_node_level_compressed(node);
}

Node* LogicalCircuit::or_node(std::vector<Node*> children) {
    Node* node = Node::createGate(Sum);
    for (auto child : children)
        node->add_child(child);
    return add_node_level_compressed(node);
}


// ---------------------------------------------------------------------------
// SDD parser
// ---------------------------------------------------------------------------

static Node* parseSDDFile(const std::string& filename, LogicalCircuit& circuit,
                           std::vector<int>& true_lits, std::vector<int>& false_lits) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    std::vector<Node*> nodeIds; // maps SDD node id -> circuit Node*
    Node* node = nullptr;

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == 'c') continue; // comment line

        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "sdd") {
            int nbNodes;
            iss >> nbNodes;
            nodeIds.resize(nbNodes);
            continue;
        }
        std::size_t nodeId;
        iss >> nodeId;

        if (type == "F") {
            node = Node::createConstant(0);
        } else if (type == "T") {
            node = Node::createConstant(1);
        } else if (type == "L") {
            int vtree, literal;
            iss >> vtree >> literal;
            // Propagate known true/false literals before adding to circuit.
            if (std::find(true_lits.begin(), true_lits.end(), literal) != true_lits.end())
                node = Node::createConstant(1);
            else if (std::find(false_lits.begin(), false_lits.end(), literal) != false_lits.end())
                node = Node::createConstant(0);
            else {
                Lit l = Lit::fromInt(literal);
                node = Node::createLeaf(l.internal_val(), mix_hash(l.internal_val()));
            }
        } else if (type == "D") {
            // Decision node: an OR over (prime AND sub) pairs.
            int vtree, numElements;
            iss >> vtree >> numElements;
            node = Node::createGate(LogicalCircuit::Sum);
            for (std::size_t i = 0; i < (std::size_t)numElements; ++i) {
                int primeId, subId;
                iss >> primeId >> subId;
                Node* and_node = Node::createGate(LogicalCircuit::Product);
                and_node->add_child(nodeIds[primeId]);
                and_node->add_child(nodeIds[subId]);
                and_node = circuit.add_node_level_compressed(and_node);
                node->add_child(and_node);
            }
        } else {
            throw std::runtime_error("Unknown node type: " + type);
        }
        node = circuit.add_node_level_compressed(node);
        nodeIds[nodeId] = node;
    }
    file.close();
    return node;
}


// ---------------------------------------------------------------------------
// D4 parser
// ---------------------------------------------------------------------------

static Node* parseD4File(const std::string& filename, LogicalCircuit& circuit,
                          std::vector<int>& true_lits, std::vector<int>& false_lits) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    // nodes[0] is a sentinel; real node ids start at 1.
    std::vector<Node*> nodes = {nullptr};
    Node* node = nullptr;

    std::string line;
    while (std::getline(file, line)) {
        // Lines starting with a node-type character declare a new node.
        switch (line[0]) {
            case 'o': node = Node::createGate(LogicalCircuit::Sum);     break;
            case 'a': node = Node::createGate(LogicalCircuit::Product); break;
            case 'f': node = Node::createConstant(0);                   break;
            case 't': node = Node::createConstant(1);                   break;
            default:  node = nullptr;
        }
        if (node != nullptr) {
            nodes.push_back(node);
        } else {
            // Edge line: "parent child lit [lit ...] 0"
            std::size_t parent, child;
            int lit;
            std::istringstream iss(line);
            iss >> parent >> child >> lit;

            // Finalize child before connecting it to its parent.
            nodes[child] = circuit.add_node_level_compressed(nodes[child]);

            if (lit == 0) {
                // Plain edge with no associated literals.
                nodes[parent]->add_child(nodes[child]);
                continue;
            }

            // Edge carries literals: wrap child + literals in a Product node.
            // If the parent is already a Product, fold directly into it to
            // avoid creating an unnecessary intermediate node.
            Node* edge;
            if (nodes[parent]->gate_type == LogicalCircuit::Product) {
                edge = nodes[parent];
            } else {
                edge = Node::createGate(LogicalCircuit::Product);
            }
            edge->add_child(nodes[child]);
            while (lit != 0) {
                // Propagate known true/false literals.
                if (std::find(true_lits.begin(), true_lits.end(), lit) != true_lits.end())
                    node = Node::createConstant(1);
                else if (std::find(false_lits.begin(), false_lits.end(), lit) != false_lits.end())
                    node = Node::createConstant(0);
                else {
                    Lit l = Lit::fromInt(lit);
                    node = Node::createLeaf(l.internal_val(), mix_hash(l.internal_val()));
                }
                edge->add_child(circuit.add_node_level_compressed(node));
                iss >> lit;
            }
            if (edge != nodes[parent]) {
                // Edge was a new intermediate node; attach it to the parent.
                edge = circuit.add_node_level_compressed(edge);
                nodes[parent]->add_child(edge);
            }
        }
    }

    // Root node (index 1) is never seen as a child, so finalize it manually.
    nodes[1] = circuit.add_node_level_compressed(nodes[1]);
    return nodes[1];
}


// ---------------------------------------------------------------------------
// Public file-loading methods
// ---------------------------------------------------------------------------

Node* LogicalCircuit::add_sdd_from_file(const std::string& filename,
                                         std::vector<int>& true_lits,
                                         std::vector<int>& false_lits) {
    Node* new_root = parseSDDFile(filename, *this, true_lits, false_lits);
    roots.push_back(new_root);
#ifndef NDEBUG
    to_dot_file(*this, "circuit_sdd.dot");
#endif
    return new_root;
}

Node* LogicalCircuit::add_d4_from_file(const std::string& filename,
                                        std::vector<int>& true_lits,
                                        std::vector<int>& false_lits) {
    Node* new_root = parseD4File(filename, *this, true_lits, false_lits);
    roots.push_back(new_root);
#ifndef NDEBUG
    to_dot_file(*this, "circuit_d4.dot");
#endif
    return new_root;
}


// ---------------------------------------------------------------------------
// Debug output
// ---------------------------------------------------------------------------

static const char* gate_label(int gate_type) {
    switch (gate_type) {
        case LogicalCircuit::Sum:     return "O";
        case LogicalCircuit::Product: return "A";
        default:                      return "G";
    }
}

static std::string semantic_label(const Node* node) {
    switch (node->type) {
        case NodeType::Constant: return node->ix == 1 ? "T" : "F";
        case NodeType::Leaf:     return "L";
        case NodeType::Gate:     return gate_label(node->gate_type);
        default:
            throw std::runtime_error("Invalid node type");
    }
}

void to_dot_file(LogicalCircuit& circuit, const std::string& filename) {
    std::ofstream file(filename);
    file << "digraph G {" << std::endl;
    for (const auto& layer : circuit.layers) {
        for (const auto* node : layer) {
            for (Node* child : node->children)
                file << "  " << child->hash << " -> " << node->hash << std::endl;
            // Label: type + layer/index for easy identification.
            file << "  " << node->hash
                 << " [label=\"" << semantic_label(node)
                 << node->layer << "/" << node->ix << "\"]" << std::endl;
        }
    }
    // Group nodes at the same layer on the same rank for a cleaner layout.
    for (const auto& layer : circuit.layers) {
        file << "  { rank=same; ";
        for (const auto* node : layer)
            file << node->hash << "; ";
        file << "}" << std::endl;
    }
    file << "}" << std::endl;
}
