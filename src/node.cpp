#include "node.h"

/**
 * Improve bit dispersion of a given hash value h.
 */
std::size_t mix_hash(std::size_t h) {
    return (h ^ (h << 16) ^ 89869747UL) * 3644798167UL;
}

/*
 * ----------
 *  Node
 * ----------
 */

/**
 * Add child to this node.
 * - Updates this.children;
 * - Updates this.hash;
 * - Increases the layer of this node to be at least above the child's layer.
 *
 * Layer parity enforcement (e.g. And=odd, Or=even) is NOT done here.
 * It is the responsibility of the caller (e.g. LogicalCircuit factory methods)
 * to call enforce_parity() after all children have been added.
 *
 * @param child The new child of this node.
 */
void Node::add_child(Node* child) {
    if (type != NodeType::Or && type != NodeType::And) {
        throw std::runtime_error("Can only add children to AND/OR nodes");
    }

    children.push_back(child);
    hash ^= mix_hash(child->hash);
    layer = std::max(layer, child->layer + 1);
}

/**
 * Useful for printing.
 * @return The label of this node.
 */
std::string Node::get_label() const {
    std::string labelName;
    switch (type) {
        case NodeType::True: labelName = "T"; break;
        case NodeType::False: labelName = "F"; break;
        case NodeType::Or: labelName = "O"; break;
        case NodeType::And: labelName = "A"; break;
        case NodeType::Leaf: labelName = "L"; break;
        default: // should not happen. Indicates node was deleted?
            throw std::runtime_error("Invalid node type");
    }
    return labelName + std::to_string(layer) + "/" + std::to_string(ix);
}


Node* Node::createLiteralNode(Lit lit) {
    int ix = lit.internal_val();
    return new Node{
            NodeType::Leaf,
            ix,
            {},
            0,
            mix_hash(ix)
    };
}

Node* Node::createAndNode() {
    return new Node{
            NodeType::And,
            -1,
            {},
            0,
            13643702618494718795UL
    };
}

Node* Node::createOrNode() {
    return new Node{
            NodeType::Or,
            -1,
            {},
            0,
            10911628454825363117UL
    };
}

Node* Node::createTrueNode() {
    return new Node{
            NodeType::True,
            1,
            {},
            0,
            10398838469117805359UL
    };
}

Node* Node::createFalseNode() {
    return new Node{
            NodeType::False,
            0,
            {},
            0,
            2055047638380880996UL
    };
}


bool compareNode(const Node& first_node, const Node& second_node) {
    return first_node.hash < second_node.hash;
}
