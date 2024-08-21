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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>

#include "node.h"
#include "hash_set8.hpp"

namespace nb = nanobind;
using namespace nb::literals;

typedef std::vector<nb::ndarray<nb::numpy, long int, nb::shape<-1>>> Arrays;


class Circuit {

public:
    // Circuit representation as a Merkle DAG
    std::vector<emhash8::HashSet<Node*, NodeHash, NodeEqual>> layers;
    // Root nodes in order they were added to the Circuit
    std::vector<Node*> roots = {};

    ~Circuit() {
        for (auto& layer: layers) {
            for (auto& node: layer)
                delete node;
            layer.clear();
        }
    }

private:

    /**
     * Add the given node to the circuit.
     *
     * Beware! This does not add intermediate nodes to intermediate layers.
     * Please use add_node_level instead.
     *
     * Uses node->layer to consider the correct layer of the circuit.
     * After adding a node, the Circuit assumes ownership and will free it upon deletion.
     * If the given node was equal to one already present, the given node is freed (deleted),
     * and the already found node is returned.
     *
     * This will update node->ix.
     *
     * @param node The node to add.
     * @return If no equivalent node was present yet, returns the node itself and true.
     * If there was already an equivalent node present, returns a pointer to that node, and false.
     */
    Node* add_node(Node* node);

    /**
     * Add a root node to the circuit, and adjust the circuit
     * so that all roots are still on the same level.
     */
    void add_root(Node* new_root, int old_depth);

public:

    /**
     * Add node to this circuit and ensure each child is in the previous adjacent layer.
     *
     * If a child does not exist in the previous layer, a chain of dummy nodes will be added in between.
     * Uses node->layer to consider the correct layer of the circuit.
     * After adding a node, the Circuit assumes ownership and will free it upon deletion.
     *
     * This may change node->children and will update node->ix.
     *
     * Importantly, we assume that the children are already part of the circuit.
     * For this reason we also return a pair, if an equivalent node (but different instance!)
     * was already present, we simply return that node and free (delete) the given node)
     *
     * @param node The new node to add to the circuit. May be freed (deleted).
     * @return If no equivalent node was present yet, returns the node itself and true.
     * If there was already an equivalent node present, returns a pointer to that node, and false.
     */
    Node* add_node_level(Node* node);

    /**
     * Get the corresponding node in the circuit.
     * This may be a different node instance with the same hash and
     * is equal according to the `NodeEqual` struct.
     */
    Node* get_node(Node* node) { return *(layers[node->layer].find(node)); }

    /**
     * Number of layers in the circuit.
     */
    inline std::size_t nb_layers() const { return layers.size(); }

    void add_SDD_from_file(const std::string &filename);

    inline std::pair<Arrays, Arrays> get_indices() { return tensorize(); }

    std::pair<Arrays, Arrays> tensorize();

    /**
     * Condition the vec of literals to be true.
     */
    void condition(const std::vector<int>& lits);

    /**
     * Number of nodes in the whole circuit.
     */
    std::size_t nb_nodes() const {
        std::size_t count = 0;
        for (const auto &layer: layers)
            count += layer.size();
        return count;
    }

    /**
     * For debugging purposes;
     * prints every node of each layer
     */
    void print_circuit() const {
        for (const auto &layer : layers) {
            std::cout << "--- next layer ---" << std::endl;
            for (const auto &node : layer) {
                std::cout << node->get_label() << " connects to ";
                for (const auto &child : node->children) {
                    std::cout << child->get_label() << ",";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

};
