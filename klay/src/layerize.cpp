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


unsigned int parseSDDFile(const std::string& filename, std::vector<Node*>& nodes) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 0;
    }

    std::vector<int> nodeIds = {};
    unsigned int maxDepth = 0;

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
            nodeIds.reserve(nbNodes);
            continue;
        }
        unsigned int nodeId;
        iss >> nodeId;

        if (type == "F") {
            nodeIds[nodeId] = nodes.size();
            nodes.push_back(createFalseNode());
        } else if (type == "T") {
            nodeIds[nodeId] = nodes.size();
            nodes.push_back(createTrueNode());
        } else if (type == "L") {
            int vtree, literal;
            iss >> vtree >> literal;
            nodeIds[nodeId] = nodes.size();
            nodes.push_back(createLiteralNode(literal));
        } else if (type == "D") {
            int vtree, numElements;
            iss >> vtree >> numElements;
            Node* or_node = createOrNode();
            for (int i = 0; i < numElements; ++i) {
                int primeId, subId;
                iss >> primeId >> subId;
                Node* and_node = createAndNode();
                and_node->add(nodes[nodeIds[primeId]]);
                and_node->add(nodes[nodeIds[subId]]);
                nodes.push_back(and_node);
                or_node->add(and_node);
            }
            nodeIds[nodeId] = nodes.size();
            nodes.push_back(or_node);
            maxDepth = std::max(maxDepth, or_node->depth);
        } else {
            std::cerr << "Could not parse: " << line << std::endl;
        }

    }
    file.close();
    return maxDepth;
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


Node* add_node(Node* node, std::unordered_map<int, Node*>& merkle) {
    if (!merkle.count(node->hash)) {
        merkle[node->hash] = node;
    }
    return merkle[node->hash];
}

void layerize(std::vector<Node*> nodes, std::unordered_map<int, Node*>& merkle) {
    // 1. Inserts the nodes in a merkle tree
    //    (layerize can be reapplied with same merkle to merge circuits)
    // 2. Assures that all children of a node have the same depth
    // 3. Assures that all nodes inner have the same arity.

    std::vector<std::size_t> arity = {};

    for (Node* node : nodes) {
        for (unsigned int i = 0; i < node->children.size(); ++i) {
            // Update pointer as the child might have been merged
            node->children[i] = merkle[node->children[i]->hash];

            // Add a chain of dummy nodes to bring child to the correct depth
            while (node->children[i]->depth < node->depth-1) {
                Node* dummy = node->children[i]->dummy_parent();
                node->children[i] = add_node(dummy, merkle);
            }
        }

        // insert node in the merkle DAG
        node = add_node(node, merkle);

        // Update the arity of the node's layer
        while (node->depth >= arity.size()) {
            arity.push_back(0);
        }
        arity[node->depth] = std::max(arity[node->depth], node->children.size());
    }


    // Add True and False to the merkle (in case they don't exist in the circuit)
    Node* true_node = add_node(createTrueNode(), merkle);
    Node* false_node = add_node(createFalseNode(), merkle);
    std::vector<Node*> neutral_elements = {true_node};

    // Make sure we have True/False in every layer (except the last)
    for (std::size_t i = 1; i < arity.size()-1; ++i) {
        true_node = add_node(true_node->dummy_parent(), merkle);
        false_node = add_node(false_node->dummy_parent(), merkle);
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
}


void tensorize(std::unordered_map<int, Node*>& merkle) {
    // Width of every layer. Width is at least 2 (for True and False nodes)
    std::vector<int> widths = {2};

    // Assign a layer index to each node
    for (const auto &[_, node]: merkle) {
        while (node->depth >= widths.size()) {
            widths.push_back(2);
        }
        if (node->type == NodeType::Leaf) {
            widths[node->depth] = std::max(widths[node->depth], node->ix + 1);
        } else if (node->ix == -1) {
            node->ix = widths[node->depth]++;
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
            if (child->ix == -1) {
                std::cerr << "[WARNING]: Node " << node->get_label()
                    << " has uninitialized child " << child->get_label() << std::endl;
                std::cerr << child->ix << " <-> " << merkle[child->hash]->ix << std::endl;
            }
            if (child->ix >= widths[child->depth]) {
                std::cerr << "[WARNING]: Node " << node->get_label()
                    << " has child " << child->get_label() << " with invalid index " << child->ix << std::endl;
            }
            layers[node->depth][node->ix].push_back(child->ix);
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
    std::vector<Node*> sdd = {};
    unsigned int sdd_depth = parseSDDFile(filename, sdd);
    // to_dot_file(map, "layerized.dot");

    std::unordered_map<int, Node*> merkle = {};
    layerize(sdd, merkle);
    tensorize(merkle);
    // to_dot_file(merkle, "tensorized.dot");
}
