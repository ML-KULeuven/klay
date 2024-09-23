/*
MIT License

Copyright (c) 2024 Anonymized

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

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>

#include "literal.h"

#include "cassert"

enum class NodeType {True, False, Or, And, Leaf};


/**
 * A Node in a Layer.
 * Sum layers are even; Product layers are odd.
 */
class Node {

public:
    NodeType type;
    int ix;  // Index of the node in its layer; can be -1 when uninitialized.

    std::list<Node*> children;
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
#ifndef NDEBUG
        // We currently assume the hash is collision-free,
        // which is a relatively safe assumption since it
        // must only be unique per layer, and the hash function
        // is relatively good.
        // This assertion checks whether we were wrong.
//        bool r = (lhs->hash == rhs->hash) && (lhs->layer == rhs->layer);
//        if (r) {
//            lhs->children.sort(compareNode); // canonical order
//            rhs->children.sort(compareNode); // canonical order
//            assert (lhs->children == rhs->children);
//        }

        // If we decide to be 100% correct; we can
        // sort the children during construction (or before adding the node).
        // and then perform the list equality check during this equality check.
        // An inductive correctness proof then follows easily.
        // If we can assume each literal and constant node have a unique hash (base-case).
        // then the invariant that the nodes in the previous layer are all unique holds,
        // and since we use equal children check, the nodes in the current layer are
        // therefore then also unique.
#endif

        // We must not compare `ix`, because that is not set yet when we compare.
        // We do not compare `type`, as that check is subsumed by comparing `layer`
        return (lhs->hash == rhs->hash) && (lhs->layer == rhs->layer);
    }
};


