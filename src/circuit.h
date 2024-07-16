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
#include "hash_table8.hpp"

namespace nb = nanobind;
using namespace nb::literals;

typedef std::vector<nb::ndarray<nb::numpy, long int, nb::shape<-1>>> Arrays;


class Circuit {

public:
    // Circuit representation as a Merkle DAG
    std::vector<emhash8::HashMap<std::size_t, Node*>> layers;

    Node* add_node(Node* node);

    /**
     * Add node to this circuit and ensure each child is in the previous adjacent layer.
     *
     * If a child is not, a chain of dummy nodes will be added in between.
     * @param node The new node to add to the circuit.
     */
    void add_node_level(Node* node);

    /**
     * Get the corresponding node in the circuit.
     * This may be a different node with the same hash.
     */
    inline Node* get_node(Node* node) {
        return layers[node->layer][node->hash];
    }

    /**
     * Number of layers in the circuit.
     */
    inline std::size_t nb_layers() const {
        return layers.size();
    }

    /**
     * Get the roots of this circuit
     */
    inline std::vector<Node*> get_roots() const {
        std::vector<Node*> roots = {};
        if (layers.size() > 0) {
            for (const auto &[_, node]: layers.back()) {
                roots.push_back(node);
            }
        }
        return roots;
    }

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

};
