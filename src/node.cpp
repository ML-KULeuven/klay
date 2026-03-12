#include "node.h"

std::size_t mix_hash(std::size_t h) {
    return (h ^ (h << 16) ^ 89869747UL) * 3644798167UL;
}

/*
 * ----------
 *  Node
 * ----------
 */

void Node::add_child(Node* child) {
    if (type != NodeType::Internal)
        throw std::runtime_error("Can only add children to Internal nodes");

    children.push_back(child);
    hash ^= mix_hash(child->hash);
    layer = std::max(layer, child->layer + 1);
}

std::string Node::get_label() const {
    std::string labelName;
    switch (type) {
        case NodeType::Internal: labelName = "I" + std::to_string(gate_type); break;
        case NodeType::Leaf:     labelName = is_constant() ? "C" : "L";       break;
        default:
            throw std::runtime_error("Invalid node type");
    }
    return labelName + std::to_string(layer) + "/" + std::to_string(ix);
}


Node* Node::createInternal(int gate_type) {
    return new Node{
        NodeType::Internal,
        gate_type,
        -1,
        {},
        0,
        13643702618494718795UL
    };
}

Node* Node::createLeaf(int ix, std::size_t hash) {
    return new Node{
        NodeType::Leaf,
        -1,
        ix,
        {},
        0,
        hash
    };
}

Node* Node::createConstant(int value) {
    // Constants use fixed hashes distinct from any mix_hash(literal) output.
    // ix == 0 → False, ix == 1 → True.
    std::size_t h = (value == 1) ? 10398838469117805359UL : 2055047638380880996UL;
    return new Node{
        NodeType::Leaf,
        -1,
        value,
        {},
        0,
        h
    };
}


bool compareNode(const Node& first_node, const Node& second_node) {
    return first_node.hash < second_node.hash;
}
