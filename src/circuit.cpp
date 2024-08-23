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

Node* Circuit::add_node(Node* node) {
    if (layers.size() <= node->layer)
        layers.resize(node->layer + 1);
    auto& layer = layers[node->layer];
    auto [it, inserted] = layer.insert(node);
    if (inserted && node->ix == -1)
        node->ix = layer.size()-1;
    if (*it != node) // did not insert; found different but equal instance
        delete node; // fix mem leak
    return *it;
}


Node* Circuit::add_node_level(Node* node) {
    // First make sure each child is adjacent.
    for (auto& child : node->children) {
#ifndef NDEBUG
        // We assume children are already part of the circuit,
        // since each child should have been added to the circuit first,
        // and they should have used the returned child.
        // It is also the user's responsibility to delete duplicate nodes
        // in case an equivalent one was already present.
        Node* child_stored = get_node(child);
        assert(child_stored == child);
#endif
        // Add a chain of dummy nodes to bring child to the correct layer
        // invariant: each child is part of the circuit.
        while (child->layer < node->layer - 1)
            child = add_node(child->dummy_parent());
    }
    // Note: since we may have changed the children, (replaced by dummy parent)
    // the hash is no longer a hash of the direct children.
    // Instead, it became the hash of the next non-dummy child.
    // As long as we are fine with the latter definition,
    // and we are consistent with that, there is no need
    // to recompute the hash of `node`.

    // Add node -- this may free node
    return add_node(node);
}

Node* Circuit::add_node_level_compressed(Node* node) {
    if (node->type != NodeType::And && node->type != NodeType::Or)
        return add_node_level(node);

    NodeType annihilateType;
    NodeType neutralType;
    Node* (*annihilate_function)();
    Node* (*neutral_function)();
    if (node->type == NodeType::Or) {
        annihilateType = NodeType::True;
        neutralType = NodeType::False;
        annihilate_function = &Node::createTrueNode;
        neutral_function = &Node::createFalseNode;
    } else if (node->type == NodeType::And) {
        annihilateType = NodeType::False;
        neutralType = NodeType::True;
        annihilate_function = &Node::createFalseNode;
        neutral_function = &Node::createTrueNode;
    } else {
        return add_node_level(node);
    }

    // Iterate over node->children
    // if child->type == neutralType
    // remove child from children.
    // if child->type == annihilateType
    // result should be true or false node (depends)
    bool annihilate = false;
    for (auto it = node->children.begin(); it != node->children.end(); ) {
        if ((*it)->type == neutralType) {
            it = node->children.erase(it);
        } else if ((*it)->type == annihilateType) {
            annihilate = true;
            break;
        } else {
            ++it;
        }
    }

    if (annihilate) { // a child was annihilating
        delete node;
        return add_node_level(annihilate_function());
    }
    if (node->children.empty()) { // all children are neutral
        delete node;
        return add_node_level(neutral_function());
    }
    if (node->children.size() == 1) {
        Node* child = node->children.front();
        delete node;
        return child;
    }

    return add_node_level(node);
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
                and_node = circuit.add_node_level_compressed(and_node);
                node->add_child(and_node);
            }
        } else {
            throw std::runtime_error("Unknown node type: " + type);
        }
        node = circuit.add_node_level_compressed(node);
        nodeIds[nodeId] = node; // Invariant: these nodes are present in the circuit.
    }
    file.close();
    return node;
}

size_t Circuit::max_layer_width() const {
    size_t max_width = 0;
    for (const auto& layer: layers)
        if (layer.size() > max_width)
            max_width = layer.size();
    return max_width;
}

void Circuit::remove_unused_nodes() {
    std::vector<std::vector<bool>> used;
    used.reserve(nb_layers());
    for (const auto& layer : layers)
        used.emplace_back(layer.size(), false);

    // set roots as used
    for (auto &root : roots)
        used[root->layer][root->ix] = true;

    // iterate backwards over layers
    // tag children of useful nodes
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        for (auto &node : *it) {
            if (used[node->layer][node->ix])
                for (auto child: node->children)
                    used[child->layer][child->ix] = true;
        }
    }

    // Now delete useless nodes (skip input)
    for (std::size_t i = 1; i < nb_layers(); ++i) {
        for (auto it = layers[i].begin(); it != layers[i].end();) {
            if (!used[i][(*it)->ix]) {
                delete *it;
                it = layers[i].erase(it);
            } else {
                ++it;
            }
        }
    }

    // Clean-up: Update ix
    for (std::size_t i = 1; i < nb_layers(); ++i) {
        assert(!layers[i].empty()); // TODO: A layer could become empty. maybe issue for tensorize()?
        int index = 0;
        for (auto &node : layers[i])
            node->ix = index++;
    }
    // Clean-up: last layer has fixed ix order
    for(size_t i = 0; i < roots.size(); ++i)
        roots[i]->ix = i;
}

Node* parseD4File(const std::string& filename, Circuit& circuit) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<Node*> nodes = {nullptr};
    Node* node;

    std::string line;
    while (std::getline(file, line)) {
        switch (line[0]) {
            // Parse new nodes
            case 'o': node = Node::createOrNode(); break;
            case 'a': node = Node::createAndNode(); break;
            case 'f': node = Node::createFalseNode(); break;
            case 't': node = Node::createTrueNode(); break;
            default: node = nullptr;
        }
        if (node != nullptr) {
            nodes.push_back(node);
        } else {
            // Parse edges
            std::size_t parent, child;
            int lit;
            std::istringstream iss(line);
            iss >> parent >> child >> lit;

            // When a child is used, we can assume it's been finalized
            nodes[child] = circuit.add_node_level_compressed(nodes[child]);
            if (lit == 0) {
                // pure edge with no associated literals
                nodes[parent]->add_child(nodes[child]);
                continue;
            }

            // edge with literals
            Node* edge;
            if (nodes[parent]->type == NodeType::And) {
                edge = nodes[parent]; // For and nodes, we can fold in the edge
            } else {
                edge = Node::createAndNode();
            }
            edge->add_child(nodes[child]);
            while (lit != 0) {
                node = Node::createLiteralNode(Lit::fromInt(lit));
                edge->add_child(circuit.add_node_level_compressed(node));
                iss >> lit;
            }
            if (edge != nodes[parent]) {
                edge = circuit.add_node_level_compressed(edge);
                nodes[parent]->add_child(edge);
            }
        }
    }

    // Root node is never used, so we need to manually add it
    nodes[1] = circuit.add_node_level_compressed(nodes[1]);
    return nodes[1];
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

void Circuit::add_root(Node* new_root, int old_depth) {
    // Bring roots to the same layer
    if (old_depth >= 0) {
        while (old_depth > new_root->layer)
            new_root = add_node(new_root->dummy_parent());

        for (; old_depth < new_root->layer; ++old_depth) {
            for (std::size_t i = 0; i < roots.size(); ++i)
                roots[i] = add_node(roots[i]->dummy_parent());
        }
    }
    roots.push_back(new_root);
    if (nb_layers() > 1)
        for(size_t i = 0; i < roots.size(); ++i)
            roots[i]->ix = i;
}


void Circuit::add_SDD_from_file(const std::string &filename) {
    int old_depth = layers.size() - 1;
    Node* new_root = parseSDDFile(filename, *this);
    add_root(new_root, old_depth);
    remove_unused_nodes();
#ifndef NDEBUG
    to_dot_file(*this, "circuit_sdd.dot");
#endif
}

void Circuit::add_D4_from_file(const std::string &filename) {
    int old_depth = layers.size() - 1;
    Node* new_root = parseD4File(filename, *this);
    add_root(new_root, old_depth);
    remove_unused_nodes();
#ifndef NDEBUG
    to_dot_file(*this, "circuit_d4.dot");
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

    if (layers.size() == 1)
        // add node for roots
        for (Node* root: roots)
            add_node(root->dummy_parent());


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



namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(nanobind_ext, m) {
m.doc() = "Layerize an SDD";

nb::class_<Circuit>(m, "Circuit")
.def(nb::init<>())
.def("add_SDD_from_file", &Circuit::add_SDD_from_file, "filename"_a)
.def("add_D4_from_file", &Circuit::add_D4_from_file, "filename"_a)
.def("get_indices", &Circuit::get_indices)
.def("condition", &Circuit::condition, "lits"_a)
.def("nb_nodes", &Circuit::nb_nodes);
}