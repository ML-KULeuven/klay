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
    // return add_node_level(node); // To disable compression.
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
    std::list<Node*> new_children = {};
    for (auto &child : node->children) {
        if (child->type == neutralType) {
            continue;
        } else if (child->type == annihilateType) {
            delete node;
            return add_node_level(annihilate_function());
        } else {
            new_children.push_back(child);
        }
    }

    if (new_children.empty()) { // all children are neutral
        delete node;
        return add_node_level(neutral_function());
    }
    if (new_children.size() == 1) {
        Node* child = new_children.front();
        delete node;
        return child;
    }
    if (new_children.size() != node->children.size()) {
        // recreate node because the hash is wrong
        // when we change children
        NodeType t = node->type;
        delete node;
        if (t == NodeType::And)
            node = Node::createAndNode();
        else
            node = Node::createOrNode();
        for(auto child: new_children)
            node->add_child(child);
    }

    return add_node_level(node);
}


Node* Circuit::add_node_merge(Node *node) {
    Node* new_node; // The hash of the first node might be spoiled
    if (node->type == NodeType::Or) {
        new_node = Node::createOrNode();
    } else if (node->type == NodeType::And) {
        new_node = Node::createAndNode();
    } else {
        return add_node_level_compressed(node);
    }

    emhash8::HashSet<Node*, NodeHash, NodeEqual> children_set = {};
    for (auto child: node->children) {
        while (child->layer < node->layer - 1)
            child = add_node(child->dummy_parent());
        children_set.insert(child);
    }
    delete node;

    for (auto child: children_set)
        new_node->add_child(child);

    return add_node_level_compressed(new_node);
}

/**
 * Auxiliary method for Circuit::add_sdd_from_file
 */
Node* parseSDDFile(const std::string& filename, Circuit& circuit, std::vector<int>& true_lits, std::vector<int>& false_lits) {
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
            if (std::find(true_lits.begin(), true_lits.end(), literal) != true_lits.end())
                node = Node::createTrueNode();
            else if (std::find(false_lits.begin(), false_lits.end(), literal) != false_lits.end())
                node = Node::createFalseNode();
            else
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
    // Should be run before adding a final root layer;
    // because it might change ix's.
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
                Node* del_node = *it;
                it = layers[i].erase(it);
                delete del_node;
            } else {
                ++it;
            }
        }
    }

    // Clean-up: last layers can be empty, pop those.
    // Intermediate layers can not be empty because we use dummy nodes.
    for (std::size_t i = nb_layers()-1; i > 0; --i) {
        if (layers[i].empty()) {
            layers.pop_back();
        } else {
            break;
        }
    }

    // Clean-up: Update ix
    for (std::size_t i = 1; i < nb_layers(); ++i) {
        // I do not think an intermediate layer can become empty.
        // However, we can have layers of only unit nodes.
        // This is an opportunity to optimise in the future.
        assert(!layers[i].empty());
        int index = 0;
        for (auto &node : layers[i])
            node->ix = index++;
    }


#ifndef NDEBUG
    // print_circuit();

    if (layers.size() > 2) {
        // check for each layer, for each node, whether the idx
        // of the children is smaller than previous layer's size.
        // We skip layer 1, because input layer does not have all nodes.
        for (std::size_t i = 2; i < nb_layers(); ++i) {
            for (auto &node: layers[i]) {
                for (auto &child: node->children) {
                    assert(child->ix < layers[i - 1].size());
                }
            }
        }
    }
#endif
}

Node* parseD4File(const std::string& filename, Circuit& circuit, std::vector<int>& true_lits, std::vector<int>& false_lits) {
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
                if (std::find(true_lits.begin(), true_lits.end(), lit) != true_lits.end())
                    node = Node::createTrueNode();
                else if (std::find(false_lits.begin(), false_lits.end(), lit) != false_lits.end())
                    node = Node::createFalseNode();
                else
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
    // Group nodes per layer
    // using { rank=same; 1; 2; } to group node 1 and 2
    for (const auto &layer: circuit.layers) {
          file << "  { rank=same; ";
          for (const auto *node : layer) {
              file << node->hash << "; ";
          }
          file << "}" << std::endl;
    }
    file << "}" << std::endl;
}

NodePtr Circuit::add_sdd_from_file(const std::string &filename, std::vector<int>& true_lits, std::vector<int>& false_lits) {
    Node* new_root = parseSDDFile(filename, *this, true_lits, false_lits);
    roots.push_back(new_root);
#ifndef NDEBUG
    to_dot_file(*this, "circuit_sdd.dot");
#endif
    return NodePtr(new_root);
}

NodePtr Circuit::add_d4_from_file(const std::string &filename, std::vector<int>& true_lits, std::vector<int>& false_lits) {
    Node* new_root = parseD4File(filename, *this, true_lits, false_lits);
    roots.push_back(new_root);
#ifndef NDEBUG
    to_dot_file(*this, "circuit_d4.dot");
#endif
    return NodePtr(new_root);
}


void Circuit::add_root_layer() {
    if (roots.empty())
        throw std::runtime_error("Cannot construct root layer, there are no roots!");

    std::size_t root_layer_index = nb_layers();
    for (std::size_t i=0; i<roots.size(); i++) {
        Node* root = roots[i];
        while (root->layer < root_layer_index) {
            root = root->dummy_parent();
            if (root->layer == root_layer_index)
                root->hash = i; // in the final layer, nodes are placed in the same order as the roots vector
            root = add_node(root);
        }
        roots[i] = root;
    }
}


void cleanup(void* data) noexcept {
	delete[] static_cast<long int*>(data);
}


std::pair<Arrays, Arrays> Circuit::get_indices() {
    remove_unused_nodes();
	add_root_layer();
    //print_circuit(); // Helpful for debugging small circuits

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


NodePtr Circuit::disjoin(std::vector<NodePtr> nodes) {
    Node* or_node = Node::createOrNode();
    for (NodePtr node_ptr: nodes) {
        Node* node = node_ptr.get();
        if (node->type == NodeType::Or) {
            for (Node* ch: node->children)
                or_node->add_child(ch);
        } else {
            or_node->add_child(node);
        }
    }
    return NodePtr(add_node_merge(or_node));
}


NodePtr Circuit::conjoin(std::vector<NodePtr> nodes) {
    Node* and_node = Node::createAndNode();
    for (NodePtr node_ptr: nodes) {
        Node* node = node_ptr.get();
        if (node->type == NodeType::And) {
            for (Node* ch: node->children)
                and_node->add_child(ch);
        } else {
            and_node->add_child(node);
        }
    }
    return NodePtr(add_node_merge(and_node));
}



NB_MODULE(nanobind_ext, m) {
m.doc() = "Layerize arithmetic circuits";

nb::class_<NodePtr>(m, "NodePtr")
.def("__repr__", &NodePtr::to_string)
.def(nb::self == nb::self)
.def("__hash__", &NodePtr::as_int)
.def("get_ix", [](NodePtr a) {return a.get()->ix;});

nb::class_<Circuit>(m, "Circuit", "Circuits are the main class added by KLay, and require no arguments to construct.\n\n:code:`circuit = klay.Circuit()` ")
.def(nb::init<>())
.def("add_sdd_from_file", &Circuit::add_sdd_from_file, "filename"_a, "true_lits"_a = std::vector<int>(), "false_lits"_a = std::vector<int>(), "Add an SDD circuit from file.\n\n:param filename:\n\tPath to the :code:`.sdd` file on disk.\n:param true_lits:\n\tList of literals that are always true and should get propagated away.\n:param false_lits:\n\tList of literals that are always false and should get propagated away.")
.def("add_d4_from_file", &Circuit::add_d4_from_file, "filename"_a, "true_lits"_a = std::vector<int>(), "false_lits"_a = std::vector<int>(), "Add an NNF circuit in the D4 format from file.\n\n:param filename:\n\tPath to the :code:`.nnf` file on disk.\n:param true_lits:\n\tList of literals that are always true and should get propagated away.\n:param false_lits:\n\tList of literals that are always false and should get propagated away.")
.def("_get_indices", &Circuit::get_indices)
.def("disjoin", &Circuit::disjoin)
.def("conjoin", &Circuit::conjoin)
.def("nb_nodes", &Circuit::nb_nodes, "Number of nodes in the circuit.")
.def("nb_root_nodes", &Circuit::nb_root_nodes, "Number of root nodes in the circuit.")
.def("true_node", &Circuit::true_node, "Adds a true node to the circuit, and returns a pointer to this node.")
.def("false_node", &Circuit::false_node, "Adds a false node to the circuit, and returns a pointer to this node.")
.def("literal_node", &Circuit::literal_node, "Adds a literal node to the circuit, and returns a pointer to this node.", "literal"_a)
.def("or_node", &Circuit::or_node, "children"_a, "Adds an :code:`or` node to the circuit, and returns a pointer to this node.")
.def("and_node", &Circuit::and_node, "children"_a, "Adds an :code:`and` node to the circuit, and returns a pointer to this node.")
.def("set_root", &Circuit::set_root, "root"_a, "Marks a node pointer as root. The order in which nodes are set as root determines the order of the output tensor. Only use this when manually constructing a circuit, when loading in a NNF/SDD its root is automatically set as root.")
.def("remove_unused_nodes", &Circuit::remove_unused_nodes, "Removes unused non-root nodes from the circuit.\nWarning: this invalidates any :code:`NodePtr` referring to an unused node (i.e., a node not connected to a root node).");

m.def("to_dot_file", &to_dot_file, "circuit"_a, "filename"_a, "Write the given circuit as dot format to a file");
}