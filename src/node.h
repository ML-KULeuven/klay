#pragma once

#include <iostream>
#include <vector>
#include <list>

#include "cassert"

/**
 * Improve bit dispersion of a given hash value h.
 */
std::size_t mix_hash(std::size_t h);

enum class NodeType {Gate, Leaf, Constant};


/**
 * A Node in a Layer.
 *
 * Structural only — no semantic types (And/Or/True/False).
 * - Gate: an inner node with children.
 * - Leaf: a terminal node (e.g., a literal).
 * - Constant: a terminal node representing a fixed value (distinguished by ix).
 */
class Node {

public:
    NodeType type;
    int gate_type; // Gate subtype (e.g., Sum=0, Product=1). Only meaningful for Gate nodes.
    int ix;  // Index of the node in its layer; can be -1 when uninitialized.

    std::list<Node*> children;
    std::size_t layer; // Layer index
    std::size_t hash; // unique identifier of the node

    static Node* createGate(int gate_type);
    static Node* createLeaf(int ix, std::size_t hash);
    static Node* createConstant(int value);

    /**
     * Add child to this node.
     * - Updates this.children;
     * - Updates this.hash;
     * - Increases the layer of this node to be at least above the child's layer.
     * @param child The new child of this node.
     */
    void add_child(Node* child);

    /**
     * Useful for printing.
     * @return The label of this node.
     */
    std::string get_label() const;

    /**
     * Whether this is a Constant node with the given value.
     */
    inline bool is_constant() const { return this->type == NodeType::Constant; }
    inline bool is_constant(int value) const { return this->type == NodeType::Constant && this->ix == value; }

};


/**
 * Binary predicate returning true if the first node goes before the second, and false otherwise.
 */
bool compareNode(const Node& first_node, const Node& second_node);


struct NodeHash {
    size_t operator()(const Node* node) const {
        return node->hash;
    }
};

struct NodeEqual {
    bool operator()(const Node* lhs, const Node* rhs) const {
        return (lhs->hash == rhs->hash) && (lhs->layer == rhs->layer);
    }
};
