#include "logical_circuit.h"
#include <cassert>
#include <fstream>
#include <sstream>
#include <algorithm>


// ---------------------------------------------------------------------------
// Internal helpers (file-local)
// ---------------------------------------------------------------------------

void cleanup(void* data) noexcept {
    delete[] static_cast<long int*>(data);
}


// ---------------------------------------------------------------------------
// LogicalCircuit
// ---------------------------------------------------------------------------

Node* LogicalCircuit::add_node_level_compressed(Node* node) {
    if (node->type != NodeType::Gate) {
        return add_node_level(node);
    }

    node->layer = resolve_layer(node->gate_type, node->layer);

    int neutral = neutral_value(node->gate_type);
    int annihilator = annihilator_value(node->gate_type);

    std::list<Node*> new_children;
    for (auto& child : node->children) {
        if (child->is_constant(neutral)) {
            continue;
        } else if (child->is_constant(annihilator)) {
            delete node;
            return add_node_level(Node::createConstant(annihilator));
        } else {
            new_children.push_back(child);
        }
    }

    if (new_children.empty()) {
        delete node;
        return add_node_level(Node::createConstant(neutral));
    }
    if (new_children.size() == 1) {
        Node* child = new_children.front();
        delete node;
        return child;
    }
    if (new_children.size() != node->children.size()) {
        // Recreate node because the hash was computed from the original children
        int gt = node->gate_type;
        delete node;
        node = Node::createGate(gt);
        for (auto child : new_children)
            node->add_child(child);
        node->layer = resolve_layer(node->gate_type, node->layer);
    }

    return add_node_level(node);
}


std::tuple<Arrays, Arrays, std::vector<int>> LogicalCircuit::get_indices() {
    remove_unused_nodes();
    add_root_layer();

    Arrays indices_ndarrays;
    Arrays csr_ndarrays;

    for (std::size_t i = 1; i < nb_layers(); ++i) {
        std::vector<long int> child_counts(layers[i].size(), 0);
        std::size_t layer_size = 0;
        std::size_t layer_len = layers[i].size() + 1;
        for (const auto* node : layers[i]) {
            layer_size += node->children.size();
            child_counts[node->ix] = node->children.size();
        }

        long int* csr_data = new long int[layer_len];
        csr_data[0] = 0;
        for (std::size_t j = 1; j < layer_len; ++j)
            csr_data[j] = csr_data[j-1] + child_counts[j-1];

        long int* indices_data = new long int[layer_size];
        for (const auto* node : layers[i]) {
            std::size_t offset = 0;
            for (Node* child : node->children) {
                assert(child->layer == i - 1);
                indices_data[csr_data[node->ix] + offset++] = child->ix;
            }
        }

        std::size_t indices_size[1] = {layer_size};
        std::size_t csr_size[1]     = {layer_len};
        nb::capsule indices_capsule(indices_data, cleanup);
        nb::capsule csr_capsule(csr_data, cleanup);

        nb::ndarray<nb::numpy, long int, nb::shape<-1>> indices_ndarray(
                indices_data, 1, indices_size, indices_capsule);
        nb::ndarray<nb::numpy, long int, nb::shape<-1>> csr_ndarray(
                csr_data, 1, csr_size, csr_capsule);
        indices_ndarrays.push_back(indices_ndarray);
        csr_ndarrays.push_back(csr_ndarray);
    }

    std::vector<int> comp_layer_types;
    for (std::size_t i = 1; i < nb_layers(); ++i)
        comp_layer_types.push_back(layer_gate_types[i]);

    return std::make_tuple(indices_ndarrays, csr_ndarrays, comp_layer_types);
}


NodePtr LogicalCircuit::true_node() {
    return NodePtr(add_node_level(Node::createConstant(1)));
}

NodePtr LogicalCircuit::false_node() {
    return NodePtr(add_node_level(Node::createConstant(0)));
}

NodePtr LogicalCircuit::literal_node(int lit) {
    if (lit == 0)
        throw std::domain_error(
            "literal_node(lit) does not allow lit == 0, "
            "because negation -0 does not make sense.");
    Lit l = Lit::fromInt(lit);
    return NodePtr(add_node_level(Node::createLeaf(l.internal_val(), mix_hash(l.internal_val()))));
}

NodePtr LogicalCircuit::and_node(std::vector<NodePtr> children) {
    Node* node = Node::createGate(Product);
    for (auto child : children)
        node->add_child(child.get());
    return NodePtr(add_node_level_compressed(node));
}

NodePtr LogicalCircuit::or_node(std::vector<NodePtr> children) {
    Node* node = Node::createGate(Sum);
    for (auto child : children)
        node->add_child(child.get());
    return NodePtr(add_node_level_compressed(node));
}


// ---------------------------------------------------------------------------
// SDD parser
// ---------------------------------------------------------------------------

static Node* parseSDDFile(const std::string& filename, LogicalCircuit& circuit,
                           std::vector<int>& true_lits, std::vector<int>& false_lits) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filename);

    std::vector<Node*> nodeIds;
    Node* node = nullptr;

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == 'c') continue;

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
            if (std::find(true_lits.begin(), true_lits.end(), literal) != true_lits.end())
                node = Node::createConstant(1);
            else if (std::find(false_lits.begin(), false_lits.end(), literal) != false_lits.end())
                node = Node::createConstant(0);
            else {
                Lit l = Lit::fromInt(literal);
                node = Node::createLeaf(l.internal_val(), mix_hash(l.internal_val()));
            }
        } else if (type == "D") {
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

    std::vector<Node*> nodes = {nullptr};
    Node* node = nullptr;

    std::string line;
    while (std::getline(file, line)) {
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
            // Parse edge: parent child lit [lit ...] 0
            std::size_t parent, child;
            int lit;
            std::istringstream iss(line);
            iss >> parent >> child >> lit;

            // Finalize child before connecting
            nodes[child] = circuit.add_node_level_compressed(nodes[child]);

            if (lit == 0) {
                nodes[parent]->add_child(nodes[child]);
                continue;
            }

            // Edge with associated literals: fold into a Product (And) node
            Node* edge;
            if (nodes[parent]->gate_type == LogicalCircuit::Product) {
                edge = nodes[parent]; // fold directly into the Product parent
            } else {
                edge = Node::createGate(LogicalCircuit::Product);
            }
            edge->add_child(nodes[child]);
            while (lit != 0) {
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
                edge = circuit.add_node_level_compressed(edge);
                nodes[parent]->add_child(edge);
            }
        }
    }

    // Root node (index 1) is never seen as a child, so finalize it manually
    nodes[1] = circuit.add_node_level_compressed(nodes[1]);
    return nodes[1];
}


// ---------------------------------------------------------------------------
// Public file-loading methods
// ---------------------------------------------------------------------------

NodePtr LogicalCircuit::add_sdd_from_file(const std::string& filename,
                                           std::vector<int>& true_lits,
                                           std::vector<int>& false_lits) {
    Node* new_root = parseSDDFile(filename, *this, true_lits, false_lits);
    roots.push_back(new_root);
#ifndef NDEBUG
    to_dot_file(*this, "circuit_sdd.dot");
#endif
    return NodePtr(new_root);
}

NodePtr LogicalCircuit::add_d4_from_file(const std::string& filename,
                                          std::vector<int>& true_lits,
                                          std::vector<int>& false_lits) {
    Node* new_root = parseD4File(filename, *this, true_lits, false_lits);
    roots.push_back(new_root);
#ifndef NDEBUG
    to_dot_file(*this, "circuit_d4.dot");
#endif
    return NodePtr(new_root);
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
            file << "  " << node->hash
                 << " [label=\"" << semantic_label(node)
                 << node->layer << "/" << node->ix << "\"]" << std::endl;
        }
    }
    for (const auto& layer : circuit.layers) {
        file << "  { rank=same; ";
        for (const auto* node : layer)
            file << node->hash << "; ";
        file << "}" << std::endl;
    }
    file << "}" << std::endl;
}
