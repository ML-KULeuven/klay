#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using namespace nb::literals;

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

#include "module.h"


enum NodeType {True, False, Or, And, Leaf};


std::size_t mix_hash(std::size_t h) {
    return (h ^ (h << 16) ^ 89869747UL) * 3644798167UL;
}

struct Node {
    NodeType type;
    int ix;  // Index of the node in its layer
    std::vector<Node*> children;
    std::size_t layer; // Layer index
    std::size_t hash; // unique identifier of the node

    void add_child(Node* child) {
        children.push_back(child);
        hash ^= mix_hash(child->hash);
        layer = std::max(layer, child->layer+1);
        if (type == NodeType::Or && layer%2 == 1) {
            std::cerr << "Sum layer " << layer << " is not even" << std::endl;
        } else if (type == NodeType::And && layer%2 == 0 ) {
            std::cerr << "Product layer " << layer << " is not odd" << std::endl;
        } else if (type != NodeType::Or && type != NodeType::And) {
            std::cerr << "Node type not recognized" << std::endl;
        }
    }

    std::string get_label() {
        std::string labelName;
        switch (type) {
            case True: labelName = "T "; break;
            case False: labelName = "F "; break;
            case Or: labelName = "O "; break;
            case And: labelName = "A "; break;
            case Leaf: labelName = "L "; break;
        }
        return labelName + std::to_string(ix);
    }

    Node* dummy_parent() {
        Node* dummy;
        if (layer % 2 == 0) {
            dummy = createAndNode();
        } else {
            dummy = createOrNode();
        }
        dummy->add_child(this);
        return dummy;
    }
};

struct Circuit {
    // Circuit representation as a Merkle DAG
    std::vector<std::unordered_map<std::size_t, Node*>> layers;

    Node* add_node(Node* node) {
        if (layers.size() <= node->layer) {
            layers.resize(node->layer + 1);
        }
        auto& layer = layers[node->layer];
        auto [it, inserted] = layer.try_emplace(node->hash, node);
        if (inserted && node->ix == -1) {
            node->ix = layer.size()-1;
        }
        if (node->children != layer[node->hash]->children) {
            std::cerr << "Hashing conflict found!!! " << node->hash << " " << layer[node->hash]->hash << std::endl;
        }

        return it->second;
    }

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

    inline Node* get_node(Node* node) {
        return layers[node->layer][node->hash];
    }

    inline std::size_t nb_layers() {
        return layers.size();
    }

    static Circuit from_SDD_file(const std::string &filename) {
        Circuit circuit;
        parseSDDFile(filename, circuit);
        // to_dot_file(circuit, "circuit.dot");
        return circuit;
    }

    std::pair<Arrays, Arrays> get_indices() {
        return tensorize(*this);
    }

    void condition(const std::vector<int>& lits) {
        for (auto &[_, node]: layers[0]) {
            if (node->type == NodeType::Leaf) {
                if (std::count(lits.begin(), lits.end(), node->ix) > 0) {
                    node->type = NodeType::True;
                    node->ix = 1;
                } else if (std::count(lits.begin(), lits.end(), -node->ix) > 0) {
                    node->type = NodeType::False;
                    node->ix = 0;
                }
            }
        }
    }

    std::size_t nb_nodes() {
        std::size_t count = 0;
        for (const auto &layer: layers) {
            count += layer.size();
        }
        return count;
    }
};


Node* createLiteralNode(int ix) {
    int i = 2 * std::abs(ix) + (ix > 0 ? 0 : 1);
    return new Node{
        NodeType::Leaf,
        i,
        {}, 0,
        mix_hash(i)
    };
}

Node* createAndNode() {
    return new Node{
        NodeType::And,
        -1, {},
        0,
        13643702618494718795UL
    };
}

Node* createOrNode() {
    return new Node{
        NodeType::Or,
        -1,
        {},
        0,
        10911628454825363117UL
    };
}

Node* createTrueNode() {
    return new Node{
        NodeType::True,
        1, {},
        0,
        10398838469117805359UL
    };
}

Node* createFalseNode() {
    return new Node{
        NodeType::False, 0,
        {}, 0,
        2055047638380880996UL
    };
}




void parseSDDFile(const std::string& filename, Circuit& circuit) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
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
        unsigned int nodeId;
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
            node->children.reserve(numElements);
            for (int i = 0; i < numElements; ++i) {
                int primeId, subId;
                iss >> primeId >> subId;
                Node* and_node = createAndNode();
                and_node->children.reserve(2);
                and_node->add_child(nodeIds[primeId]);
                and_node->add_child(nodeIds[subId]);
                circuit.add_node_level(and_node);
                node->add_child(and_node);
            }
        } else {
            std::cerr << "Could not parse: " << line << std::endl;
            return;
        }
        circuit.add_node_level(node);
        nodeIds[nodeId] = node;
    }
    file.close();
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


std::pair<Arrays, Arrays> tensorize(Circuit& circuit) {
    Arrays indices(circuit.nb_layers());
    Arrays csr(circuit.nb_layers());

    for (std::size_t i = 1; i < circuit.nb_layers(); ++i) {
        std::vector<long int> child_counts(circuit.layers[i].size(), 0);
        std::size_t layer_size = 0;
        for (const auto &[_, node]: circuit.layers[i]) {
            layer_size += node->children.size();
            child_counts[node->ix] = node->children.size();
        }

        csr[i] = std::vector<long int>(circuit.layers[i].size()+1, 0);
        for (std::size_t j = 1; j < csr[i].size(); ++j) {
            csr[i][j] = csr[i][j-1] + child_counts[j-1];
        }

        indices[i] = std::vector<long int>(layer_size, -1);
        for (const auto &[_, node]: circuit.layers[i]) {
            std::size_t offset = 0;
            for (Node *child: node->children) {
                indices[i][csr[i][node->ix] + offset++] = child->ix;
            }
        }
    }
    return std::make_pair(indices, csr);
}


NB_MODULE(nanobind_ext, m) {
    m.doc() = "Layerize an SDD";

    nb::class_<Circuit>(m, "Circuit")
            .def_static("from_SDD_file", &Circuit::from_SDD_file)
            .def("get_indices", &Circuit::get_indices)
            .def("condition", &Circuit::condition)
            .def("nb_nodes", &Circuit::nb_nodes);
}
