#pragma once

#include <iostream>
#include <vector>
#include <list>

/**
 * Improve bit dispersion of a given hash value h.
 */
std::size_t mix_hash(std::size_t h);

enum class NodeType { Internal, Leaf };


/**
 * A Node in a Layer.
 *
 * Structural only — no semantic types (And/Or/True/False).
 * - Internal: an inner node with children, further typed by gate_type (e.g. Sum=0, Product=1, -1=passthrough).
 * - Leaf: a terminal node with no children. Literals and constants are both leaves;
 *         constants are distinguished by ix == 0 (false) or ix == 1 (true),
 *         which cannot collide with literals since variable 0 is forbidden.
 */
class Node {

public:
    NodeType type;
    int gate_type; // Subtype for Internal nodes (e.g. Sum=0, Product=1, -1=passthrough). Unused for Leaf.
    int ix;        // Index within the layer. For Leaf nodes also encodes identity (literal or constant value).

    std::list<Node*> children;
    std::size_t layer;
    std::size_t hash;

    static Node* createInternal(int gate_type);
    static Node* createLeaf(int ix, std::size_t hash);

    /**
     * Create a constant (True or False) leaf node.
     * value == 1 → True, value == 0 → False.
     * Constants are Leaf nodes with ix == value and a fixed hash.
     */
    static Node* createConstant(int value);

    /**
     * Add child to this node.
     * - Updates children, hash, and layer.
     */
    void add_child(Node* child);

    /**
     * Useful for printing.
     */
    std::string get_label() const;

    inline bool is_constant() const { return type == NodeType::Leaf && ix <= 1; }
    inline bool is_constant(int value) const { return type == NodeType::Leaf && ix == value && ix <= 1; }

};


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
