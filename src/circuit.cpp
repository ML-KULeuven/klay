/*
MIT License

Copyright (c) 2024 KU Leuven (Machine Learning lab)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "circuit.h"
#include "cassert"
#define NDEBUG // Comment-out to enable asserts


std::pair<Node*, bool> Circuit::add_node(Node* node) {
    if (layers.size() <= node->layer)
        layers.resize(node->layer + 1);
    auto& layer = layers[node->layer];
    auto [it, inserted] = layer.insert(node);
    if (inserted && node->ix == -1)
        node->ix = layer.size()-1;
    return {*it, inserted };
}


std::pair<Node*, bool> Circuit::add_node_level(Node* node) {
    // First make sure each child is adjacent.
    for (auto& child : node->children) {
#ifndef NDEBUG
        // We assume children are already part of the circuit,
        // since each child should have been added to the circuit first,
        // and they should have used the returned child.
        // It is also the user's responsibility to delete duplicate nodes
        // in case an equivalent one was already present.
        Node* child_stored = get_node(child);
        assert(*child_stored == *child);
#endif
        // Add a chain of dummy nodes to bring child to the correct layer
        // invariant: each child is part of the circuit.
        bool child_inserted = false;
        while (child->layer < node->layer - 1) {
            Node* parent = child->dummy_parent();
            [child, parent_inserted] = add_node(parent);
            if (!parent_inserted) // maintain invariant.
                delete parent;

        }
    }
    //TODO: recompute hash of node! bc children hashes have changed due to multiple layer!
    return add_node(node);
}

/**
 * Auxiliary method for Circuit::add_SDD_from_file
 */
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
            node = Node::createFalseNode();
        } else if (type == "T") {
            node = Node::createTrueNode();
        } else if (type == "L") {
            int vtree, literal;
            iss >> vtree >> literal;
            node = Node::createLiteralNode(Lit::fromInt(literal));
        } else if (type == "D") {
            int vtree, numElements;
            iss >> vtree >> numElements;
            node = Node::createOrNode();
            for (std::size_t i = 0; i < numElements; ++i) {
                int primeId, subId;
                iss >> primeId >> subId;
                Node* and_node = Node::createAndNode();
                and_node->add_child(nodeIds[primeId]);
                and_node->add_child(nodeIds[subId]);
                auto [new_and_node, inserted] = circuit.add_node_level(and_node);
                if (!inserted) // remove and_node if equivalent was present already
                    delete and_node;
                and_node = new_and_node;
                node->add_child(and_node);
            }
        } else {
            throw std::runtime_error("Unknown node type: " + type);
        }
        auto [new_node, inserted] = circuit.add_node_level(node);
        if (!inserted) // remove node if equivalent was present already
            delete node;
        node = new_node;
        nodeIds[nodeId] = node; // Invariant: these nodes are present in the circuit.
    }
    file.close();
    return node;
}

/**
 * Write the given circuit as dot format to a file.
 * @param circuit The circuit to write as dot format.
 * @param filename The filepath to write to.
 */
void to_dot_file(Circuit& circuit, const std::string& filename) {
    std::ofstream file(filename);
    file << "digraph G {" << std::endl;
    for (const auto &layer: circuit.layers) {
        for (const auto *node : layer) {
            for (Node *child: node->children) {
                file << "  " << child->hash << " -> " << node->hash << std::endl;
            }
            file << "  " << node->hash << " [label=\"" << node->get_label() << "\"]" << std::endl;
        }
    }
    file << "}" << std::endl;
}

void Circuit::add_SDD_from_file(const std::string &filename) {
    int depth = layers.size() - 1;
    Node* new_root = parseSDDFile(filename, *this);

    // Bring roots to the same layer
    if (depth >= 0) {
        while (depth > new_root->layer) {
            new_root = add_node(new_root->dummy_parent()).first;
        }
        for (; depth < new_root->layer; ++depth) {
            for (std::size_t i = 0; i < roots.size(); ++i) {
                roots[i] = add_node(roots[i]->dummy_parent()).first;
            }
        }
    }
    roots.push_back(new_root);
    for(size_t i = 0; i < roots.size(); ++i) {
        roots[i]->ix = i;
    }
#ifndef NDEBUG
    to_dot_file(*this, "circuit.dot");
#endif
}

/**
 * Condition the vec of literals to be true.
 */
void Circuit::condition(const std::vector<int>& lits) {
    std::vector<Lit> lits_formatted;
    lits_formatted.reserve(lits.size());
    for (auto lit: lits)
        lits_formatted.emplace_back(Lit::fromInt(lit));
    // condition
    for (auto *node: layers[0]) {
        if (node->type == NodeType::Leaf) {
            if (std::find(lits_formatted.begin(), lits_formatted.end(), Lit(node->ix)) != lits_formatted.end()) {
                node->type = NodeType::True;
                node->ix = 1;
            } else if (std::find(lits_formatted.begin(), lits_formatted.end(), Lit(node->ix)) != lits_formatted.end()) {
                node->type = NodeType::False;
                node->ix = 0;
            }
        }
    }
}


void cleanup(void* data) noexcept {
    delete[] static_cast<long int*>(data);
}


std::pair<Arrays, Arrays> Circuit::tensorize() {
    // print_circuit(); // Helpful for debugging small circuits
    // per layer, a vector of size the number of children (but children can count twice
    // so this might be larger than simply the previous layer.
    Arrays indices_ndarrays;
    // per layer, a vector representing the layer
    Arrays csr_ndarrays;

    for (std::size_t i = 1; i < nb_layers(); ++i) {
        std::vector<long int> child_counts(layers[i].size(), 0);
        std::size_t layer_size = 0;
        std::size_t layer_len = layers[i].size()+1;
        for (const auto *node: layers[i]) {
            layer_size += node->children.size();
            child_counts[node->ix] = node->children.size();
        }

        long int* csr_data = new long int[layer_len];
        csr_data[0] = 0;
        for (std::size_t j = 1; j < layer_len; ++j) {
            csr_data[j] = csr_data[j-1] + child_counts[j-1];
        }

        long int* indices_data = new long int[layer_size];
        for (const auto *node: layers[i]) {
            std::size_t offset = 0;
            for (Node *child: node->children) {
                assert(child->layer == i-1);
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