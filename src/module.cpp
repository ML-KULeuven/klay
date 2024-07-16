#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

using namespace nb::literals;

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>

#include "module.h"
#include "hash_table8.hpp"

enum class NodeType {True, False, Or, And, Leaf};


std::size_t mix_hash(std::size_t h) {
    return (h ^ (h << 16) ^ 89869747UL) * 3644798167UL;
}

/**
 * A Node in a Layer.
 * Sum layers are even; Product layers are odd.
 */
struct Node {
    NodeType type;
    int ix;  // Index of the node in its layer
    std::list<Node*> children;
    std::size_t layer; // Layer index
    std::size_t hash; // unique identifier of the node

    /**
     * Add child to this node.
     * - Updates this.children;
     * - Updates this.hash;
     * - Increases the layer of this node to be at least above the child's layer.
     * @param child The new child of this node.
     */
    void add_child(Node* child) {
        children.push_back(child);
        hash ^= mix_hash(child->hash);
        layer = std::max(layer, child->layer+1);
        if (type == NodeType::Or && layer%2 == 1) {
            throw std::runtime_error("Sum layer is not even");
        } else if (type == NodeType::And && layer%2 == 0 ) {
            throw std::runtime_error("Product layer is not odd");
        } else if (type != NodeType::Or && type != NodeType::And) {
            throw std::runtime_error("Node type not recognized");
        }
    }

    std::string get_label() const {
        std::string labelName;
        switch (type) {
            case NodeType::True: labelName = "T "; break;
            case NodeType::False: labelName = "F "; break;
            case NodeType::Or: labelName = "O "; break;
            case NodeType::And: labelName = "A "; break;
            case NodeType::Leaf: labelName = "L "; break;
        }
        return labelName + std::to_string(ix);
    }

    /**
     * Create a dummy parent who is one layer above this node.
     * This is needed to create a chain of dummy nodes such
     * that each node only has children in the previous adjacent layer.
     * @return The dummy parent.
     */
    Node* dummy_parent() {
        Node* dummy = (layer % 2 == 0) ? createAndNode() : createOrNode();
        dummy->add_child(this);
        return dummy;
    }
};

struct Circuit {
    // Circuit representation as a Merkle DAG
    std::vector<emhash8::HashMap<std::size_t, Node*>> layers;

    Node* add_node(Node* node) {
        if (layers.size() <= node->layer) {
            layers.resize(node->layer + 1);
        }
        auto& layer = layers[node->layer];
        auto [it, inserted] = layer.try_emplace(node->hash, node);
        if (inserted && node->ix == -1) {
            node->ix = layer.size()-1;
        }
        // if (node->children != layer[node->hash]->children) {
        //     throw std::runtime_error("Hashing conflict found!!!");
        // }

        return it->second;
    }

    /**
     * Add node to this circuit and ensure each child is in the previous adjacent layer.
     *
     * If a child is not, a chain of dummy nodes will be added in between.
     * @param node The new node to add to the circuit.
     */
    void add_node_level(Node* node) {
        for (auto& child : node->children) {
            // Update pointer as the child might have been merged
            child = get_node(child);

            // Add a chain of dummy nodes to bring child to the correct layer
            while (child->layer < node->layer - 1) {
                child = add_node(child->dummy_parent());
            }
        }
        add_node(node);
    }

    /**
     * Get the corresponding node in the circuit.
     * This may be a different node with the same hash.
     */
    inline Node* get_node(Node* node) {
        return layers[node->layer][node->hash];
    }

    /**
     * Number of layers in the circuit.
     */
    inline std::size_t nb_layers() const {
        return layers.size();
    }

    /**
     * Get the roots of this circuit
     */
    inline std::vector<Node*> get_roots() const {
        std::vector<Node*> roots = {};
        if (layers.size() > 0) {
            for (const auto &[_, node]: layers.back()) {
                roots.push_back(node);
            }
        }
        return roots;
    }

    void add_SDD_from_file(const std::string &filename) {
        std::vector<Node*> roots = get_roots();
        int depth = layers.size() - 1;
        Node* new_root = parseSDDFile(filename, *this);

        // Bring roots to the same layer
        if (depth >= 0) {
            while (depth > new_root->layer) {
                new_root = add_node(new_root->dummy_parent());
            }
            for (; depth < new_root->layer; ++depth) {
                for (std::size_t i = 0; i < roots.size(); ++i) {
                    roots[i] = add_node(roots[i]->dummy_parent());
                }
            }
        }
        to_dot_file(*this, "circuit.dot");
    }

    std::pair<Arrays, Arrays> get_indices() {
        return tensorize(*this);
    }

    /**
     * Condition the vec of literals to be true.
     */
    void condition(const std::vector<int>& lits) {
        for (auto &[_, node]: layers[0]) {
            if (node->type == NodeType::Leaf) {
                if (std::find(lits.begin(), lits.end(), node->ix) != lits.end()) {
                    node->type = NodeType::True;
                    node->ix = 1;
                } else if (std::find(lits.begin(), lits.end(), -node->ix) != lits.end()) {
                    node->type = NodeType::False;
                    node->ix = 0;
                }
            }
        }
    }

    /**
     * Number of nodes in the whole circuit.
     */
    std::size_t nb_nodes() const {
        std::size_t count = 0;
        for (const auto &layer: layers) {
            count += layer.size();
        }
        return count;
    }
};


inline Node* createLiteralNode(int lit) {
    int ix = (std::abs(lit) << 1) + (lit <= 0);
    return new Node{
        NodeType::Leaf,
        ix,
        {},
        0,
        mix_hash(ix)
    };
}

inline Node* createAndNode() {
    return new Node{
        NodeType::And,
        -1,
        {},
        0,
        13643702618494718795UL
    };
}

inline Node* createOrNode() {
    return new Node{
        NodeType::Or,
        -1,
        {},
        0,
        10911628454825363117UL
    };
}

inline Node* createTrueNode() {
    return new Node{
        NodeType::True,
        1,
        {},
        0,
        10398838469117805359UL
    };
}

inline Node* createFalseNode() {
    return new Node{
        NodeType::False,
        0,
        {},
        0,
        2055047638380880996UL
    };
}




Node* parseSDDFile(const std::string& filename, Circuit& circuit) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<Node*> nodeIds = {};
    Node* node;

    std::string line;
    while (std::getline(file, line)) {
        // Ignore comment lines
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
            node = createFalseNode();
        } else if (type == "T") {
            node = createTrueNode();
        } else if (type == "L") {
            int vtree, literal;
            iss >> vtree >> literal;
            node = createLiteralNode(literal);
        } else if (type == "D") {
            int vtree, numElements;
            iss >> vtree >> numElements;
            node = createOrNode();
            for (std::size_t i = 0; i < numElements; ++i) {
                int primeId, subId;
                iss >> primeId >> subId;
                Node* and_node = createAndNode();
                and_node->add_child(nodeIds[primeId]);
                and_node->add_child(nodeIds[subId]);
                circuit.add_node_level(and_node);
                node->add_child(and_node);
            }
        } else {
            throw std::runtime_error("Unknown node type: " + type);
        }
        circuit.add_node_level(node);
        nodeIds[nodeId] = node;
    }
    file.close();
    return node;
}


void to_dot_file(Circuit& circuit, const std::string& filename) {
    std::ofstream file(filename);
    file << "digraph G {" << std::endl;
    for (const auto &layer: circuit.layers) {
        for (const auto &[_, node]: layer) {
            for (Node *child: node->children) {
                file << "  " << child->hash << " -> " << node->hash << std::endl;
            }
            file << "  " << node->hash << " [label=\"" << node->get_label() << "\"]" << std::endl;
        }
    }
    file << "}" << std::endl;
}


void cleanup(void* data) noexcept {
    delete[] static_cast<long int*>(data);
}


std::pair<Arrays, Arrays> tensorize(Circuit& circuit) {
    // per layer, a vector of size the number of children (but children can count twice
    // so this might be larger than simply the previous layer.
    Arrays indices_ndarrays;
    // per layer, a vector representing the layer
    Arrays csr_ndarrays;

    for (std::size_t i = 1; i < circuit.nb_layers(); ++i) {
        std::vector<long int> child_counts(circuit.layers[i].size(), 0);
        std::size_t layer_size = 0;
        std::size_t layer_len = circuit.layers[i].size()+1;
        for (const auto &[_, node]: circuit.layers[i]) {
            layer_size += node->children.size();
            child_counts[node->ix] = node->children.size();
        }

        long int* csr_data = new long int[layer_len];
        csr_data[0] = 0;
        for (std::size_t j = 1; j < layer_len; ++j) {
            csr_data[j] = csr_data[j-1] + child_counts[j-1];
        }

        long int* indices_data = new long int[layer_size];
        for (const auto &[_, node]: circuit.layers[i]) {
            std::size_t offset = 0;
            for (Node *child: node->children) {
                indices_data[csr_data[node->ix] + offset++] = child->ix;
            }
        }

        std::size_t indices_size[1] = {layer_size};
        std::size_t csr_size[1] = {layer_len};
        nb::capsule indices_capsule(indices_data, cleanup);
        nb::capsule csr_capsule(csr_data, cleanup);

        nb::ndarray<nb::numpy, long int, nb::shape<-1>> indices_ndarray(indices_data, 1, indices_size, indices_capsule);
        nb::ndarray<nb::numpy, long int, nb::shape<-1>> csr_ndarray(csr_data, 1, csr_size, csr_capsule);
        indices_ndarrays.push_back(indices_ndarray);
        csr_ndarrays.push_back(csr_ndarray);
    }

    return std::make_pair(indices_ndarrays, csr_ndarrays);
}


NB_MODULE(nanobind_ext, m) {
    m.doc() = "Layerize an SDD";

    nb::class_<Circuit>(m, "Circuit")
        .def(nb::init<>())
        .def("add_SDD_from_file", &Circuit::add_SDD_from_file)
        .def("get_indices", &Circuit::get_indices)
        .def("condition", &Circuit::condition)
        .def("nb_nodes", &Circuit::nb_nodes);
}
