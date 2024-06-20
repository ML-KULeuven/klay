#include "layerize.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>


enum NodeType {True, False, Or, And, Leaf};

struct Node {
    NodeType type;
    int ix;  // Index of the node in its layer
    std::vector<Node*> children;
    unsigned int depth; // Layer index
    long hash; // unique identifier of the node

    void add(Node* child) {
        children.push_back(child);
        hash += std::hash<std::string>{}(std::to_string(child->hash));
        if (child->depth + 1 > depth) {
            depth = child->depth+1;
        }
        if (type == NodeType::Or) {
            assert(depth%2 == 0);
        } else if (type == NodeType::And) {
            assert(depth%2 == 1);
        } else {
            assert(false);
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
        if (depth % 2 == 0) {
            dummy = createAndNode();
        } else {
            dummy = createOrNode();
        }
        dummy->add(this);
        if (ix == 0 || ix == 1) {
            // force the True/False nodes to be first in each layer
            dummy->ix = ix;
        } else {
            dummy->ix = -1;
        }
        return dummy;
    }
};

Node* createLiteralNode(int ix) {
    Node* node = new Node();
    node->type = NodeType::Leaf;
    node->ix = 2*std::abs(ix) + (ix > 0 ? 0 : 1);
    node->hash = std::hash<std::string>{}("L" + std::to_string(ix));
    return node;
}

Node* createAndNode() {
    Node* node = new Node();
    node->type = NodeType::And;
    node->hash = std::hash<std::string>{}("And");
    node->ix = -1;
    return node;
}

Node* createOrNode() {
    Node* node = new Node();
    node->type = NodeType::Or;
    node->hash = std::hash<std::string>{}("Or");
    node->ix = -1;
    return node;
}

Node* createTrueNode() {
    Node* node = new Node();
    node->type = NodeType::True;
    node->hash = std::hash<std::string>{}("True");
    node->ix = 1;
    return node;
}

Node* createFalseNode() {
    Node* node = new Node();
    node->type = NodeType::False;
    node->hash = std::hash<std::string>{}("False");
    return node;
}


void parseSDDFile(const std::string& filename, std::unordered_map<int, Node*>& nodes) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Ignore comment lines
        if (line[0] == 'c') continue;

        std::istringstream iss(line);
        std::string type;
        iss >> type;

        if (type == "sdd") {
            // This line contains the count of SDD nodes, which we can ignore for now.
            continue;
        }
        unsigned int nodeId;
        iss >> nodeId;

        if (type == "F") {
            nodes[nodeId] = createFalseNode();
        } else if (type == "T") {
            nodes[nodeId] = createTrueNode();
        } else if (type == "L") {
            int vtree, literal;
            iss >> vtree >> literal;
            Node* leafNode = createLiteralNode(literal);
            nodes[nodeId] = leafNode;
        } else if (type == "D") {
            int vtree, numElements;
            iss >> vtree >> numElements;
            Node* or_node = createOrNode();
            for (int i = 0; i < numElements; ++i) {
                int primeId, subId;
                iss >> primeId >> subId;
                Node* and_node = createAndNode();
                and_node->add(nodes[primeId]);
                and_node->add(nodes[subId]);
                nodes[and_node->hash] = and_node;
                or_node->add(and_node);
            }
            nodes[nodeId] = or_node;
        } else {
            std::cerr << "Could not parse: " << line << std::endl;
        }

    }
    file.close();
}


void to_dot_file(std::unordered_map<int, Node*>& circuit, const std::string& filename) {
    std::ofstream
    file(filename);
    file << "digraph G {" << std::endl;
    for (const auto& [_, node] : circuit) {
        for (Node *child: node->children) {
            file << "  " << child->hash << " -> " << node->hash << std::endl;
        }
        file << "  " << node->hash << " [label=\"" << node->get_label() << "\"]" << std::endl;
    }
    file << "}" << std::endl;
}


void layerize(Node* node, std::unordered_set<int>& visited, std::unordered_map<int, Node*>& merkle) {
    // 1. Inserts a node in a merkle tree (function can be reapplied to merge circuits)
    // 2. Assures that all children of a node have the same depth

    if (visited.count(node->hash)) {
        return;
    }
    visited.insert(node->hash);

    for (unsigned int i = 0; i < node->children.size(); ++i) {
        layerize(node->children[i], visited, merkle);
        // Update pointer as the child might have been merged
        node->children[i] = merkle[node->children[i]->hash];
        while (node->children[i]->depth < node->depth-1) {
            Node* dummy = node->children[i]->dummy_parent();
            merkle[dummy->hash] = dummy;
            node->children[i] = dummy;
        }
    }
    merkle[node->hash] = node;
}


void tensorize(std::unordered_map<int, Node*>& merkle) {
    // Width of every layer. Width is at least 2 (for True and False nodes)
    std::vector<int> widths = {2};
    // Arity of every layer. The first layer is always 0 (as these are leaf nodes)
    std::vector<int> arity = {0};

    // Assign a layer index to each node
    for (const auto &[_, node]: merkle) {
        while (node->depth >= widths.size()) {
            widths.push_back(2);
            arity.push_back(0);
        }

        if (node->type == NodeType::Leaf) {
            widths[node->depth] = std::max(widths[node->depth], node->ix + 1);
        } else if (node->ix == -1) {
            node->ix = widths[node->depth];
            widths[node->depth] += 1;
            arity[node->depth] = std::max(arity[node->depth], (int)node->children.size());
        }
    }

    // Add chains for the True and False nodes
    Node* true_node = createTrueNode();
    merkle[true_node->hash] = true_node;
    Node* false_node = createFalseNode();
    merkle[false_node->hash] = false_node;
    std::vector<Node*> neutral_elements = {true_node};

    for (unsigned int i = 1; i < widths.size()-1; ++i) {
        true_node = true_node->dummy_parent();
        false_node = false_node->dummy_parent();
        merkle[true_node->hash] = true_node;
        merkle[false_node->hash] = false_node;
        if (i % 2 == 0) {
            neutral_elements.push_back(true_node);
        } else {
            neutral_elements.push_back(false_node);
        }
    }

    // Fill up the arity of the nodes with neutral elements
    for (const auto &[_, node]: merkle) {
        if (node->type == NodeType::And || node->type == NodeType::Or) {
            for (int i = node->children.size(); i < arity[node->depth]; ++i) {
                node->children.push_back(neutral_elements[node->depth-1]);
            }
        }
    }

    // Create the tensors
    std::vector<std::vector<std::vector<int>>> layers = {};
    for (unsigned int i = 0; i < widths.size(); ++i) {
        std::vector<std::vector<int>> layer = {};
        for (int j = 0; j < widths[i]; ++j) {
            std::vector<int> node = {};
            layer.push_back(node);
        }
        layers.push_back(layer);
    }
    for (const auto& [_, node] : merkle) {
        for (Node* child : node->children) {
            // TODO: fix
            layers[node->depth][node->ix].push_back(merkle[child->hash]->ix);
        }
    }

    // Write the tensors to a file
    std::ofstream file("tensors.txt");
    for (unsigned int i = 1; i < layers.size(); ++i) {
        file << "Layer" << std::endl;
        for (unsigned int j = 0; j < layers[i].size(); ++j) {
            for (unsigned int k = 0; k < layers[i][j].size(); ++k) {
                file << layers[i][j][k] << " ";
            }
            if (layers[i][j].size() > 0) {
                // for the last layer, we don't want a newline
                file << std::endl;
            }
        }
    }
}


void brr(const std::string &filename) {
    std::unordered_map<int, Node*> map = {};
    parseSDDFile(filename, map);
    Node* root = map[0];
    // to_dot_file(map, "layerized.dot");

    std::unordered_map<int, Node*> merkle = {};
    std::unordered_set<int> visited = {};
    layerize(root, visited, merkle);
    tensorize(merkle);
    // to_dot_file(merkle, "tensorized.dot");
}
