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

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>

enum class NodeType {True, False, Or, And, Leaf};


/**
 * A Node in a Layer.
 * Sum layers are even; Product layers are odd.
 */
class Node {

public:
    NodeType type;
    int ix;  // Index of the node in its layer

    std::list<Node*> children;
    std::size_t layer; // Layer index
    std::size_t hash; // unique identifier of the node


    //Node(NodeType type, std::size_t hash, std::size_t layer, int ix) : type(type), hash(hash), layer(layer), ix(ix) {
    //}

    static Node* createLiteralNode(int lit);
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
};

