#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "literal.h"

#include "cassert"

namespace klay {

enum class NodeType {True, False, Or, And, Leaf};


/**
 * A Node in a Layer.
 * Sum layers are even; Product layers are odd.
 */
class Node {

public:
    NodeType type;
    int ix;  // Index of the node in its layer; can be -1 when uninitialized.

    std::vector<Node*> children;
    std::size_t layer; // Layer index
    std::size_t hash; // unique identifier of the node


    static Node* createLiteralNode(Lit lit);
    static Node* createOrNode();
    static Node* createAndNode();
    static Node* createTrueNode();
    static Node* createFalseNode();

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
     * Create a dummy parent who is one layer above this node.
     * This is needed to create a chain of dummy nodes such
     * that each node only has children in the previous adjacent layer.
     * @return The dummy parent.
     */
    Node* dummy_parent();

    /**
     * Whether this is a True Node.
     */
    inline bool is_true() const { return this->type == NodeType::True; }

    /**
     * Whether this is a False Node.
     */
    inline bool is_false() const { return this->type == NodeType::False; }

};


/**
 * Binary predicate returning true if the first node goes before the second, and false otherwise.
 * This can be used to sort a list of nodes.
 * Like this: `my_list.sort(compareNode);`
 * @param first_node The first node
 * @param second_node The second node
 * @return Whether the first node goes before the second.
 */
bool compareNode(const Node& first_node, const Node& second_node);


/**
 * Used to get the hash from a node.
 * For example,
 * ```
 *     emhash8::HashSet<Node*, NodeHash, NodeEqual> my_set;
 *     creates a hashSet of nodes, using this particular hash method.
 * ```
 */
struct NodeHash {
    size_t operator()(const Node* node) const {
        return node->hash;
    }
};

struct NodeEqual {
    bool operator()(const Node* lhs, const Node* rhs) const {
        // We must not compare `ix` for inner nodes, because that is not set yet when we compare.
        if (lhs->hash != rhs->hash || lhs->layer != rhs->layer || lhs->type != rhs->type)
            return false;
        if (lhs->type == NodeType::Leaf)
            return lhs->ix == rhs->ix;  // for leaves, `ix` holds the literal
        if (lhs->children.size() != rhs->children.size())
            return false;
        if (lhs->children == rhs->children)  // fast path: same children in the same order
            return true;
        // Children are unordered and may contain repeats: compare as multisets.
        std::vector<const Node*> l(lhs->children.begin(), lhs->children.end());
        std::vector<const Node*> r(rhs->children.begin(), rhs->children.end());
        std::sort(l.begin(), l.end());
        std::sort(r.begin(), r.end());
        return l == r;
    }
};


}  // namespace klay
