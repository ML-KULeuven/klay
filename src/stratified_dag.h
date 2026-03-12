#pragma once

#include <vector>
#include <list>
#include <iostream>
#include <cassert>
#include <stdexcept>

#include "node.h"
#include "hash_set8.hpp"

/**
 * A generic layered DAG (Directed Acyclic Graph) stored as a vector of hash-set layers.
 *
 * Nodes are deduplicated by (hash, layer) — two nodes with the same hash in the same
 * layer are considered equal, and only one instance is kept.
 *
 * Children of a node must reside in the immediately adjacent lower layer. If they do
 * not, `add_node_level` inserts passthrough ("dummy") Gate nodes to bridge the gap.
 */
class StratifiedDag {
public:
    // Circuit representation as a Merkle DAG
    std::vector<emhash8::HashSet<Node*, NodeHash, NodeEqual>> layers;
    // Root nodes in order they were added
    std::vector<Node*> roots;
    // Assigned gate type per layer (-1 = unassigned, e.g. leaf layer or not yet claimed)
    std::vector<int> layer_gate_types;

    ~StratifiedDag() {
        for (auto& layer : layers) {
            for (auto& node : layer)
                delete node;
            layer.clear();
        }
    }

    /**
     * Get the canonical node stored in the circuit that is equal to the given node.
     */
    Node* get_node(Node* node) { return *(layers[node->layer].find(node)); }

    inline std::size_t nb_layers() const { return layers.size(); }

    std::size_t max_layer_width() const {
        std::size_t max_width = 0;
        for (const auto& layer : layers)
            if (layer.size() > max_width)
                max_width = layer.size();
        return max_width;
    }

    std::size_t nb_nodes() const {
        std::size_t count = 0;
        for (const auto& layer : layers)
            count += layer.size();
        return count;
    }

    std::size_t nb_root_nodes() const { return roots.size(); }

    void set_root(Node* root) {
        roots.push_back(root);
    }

    /**
     * For debugging: print every node of each layer.
     */
    void print_circuit() const {
        for (const auto& layer : layers) {
            std::cout << "--- next layer ---" << std::endl;
            for (const auto& node : layer) {
                std::cout << node->get_label() << " connects to ";
                for (const auto& child : node->children)
                    std::cout << child->get_label() << ",";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    /**
     * Add node to this circuit and ensure each child is in the previous adjacent layer.
     *
     * If a child is not in the previous layer, a chain of passthrough Gate nodes is
     * inserted to bridge the gap.
     *
     * `node->layer` must already reflect the correct target layer before calling this.
     * May change `node->children` (replacing skipped children with their dummy parents)
     * and will update `node->ix`.
     *
     * If an equivalent node already exists, the given node is freed and the existing
     * node is returned.
     */
    Node* add_node_level(Node* node) {
        for (auto& child : node->children) {
#ifndef NDEBUG
            Node* child_stored = get_node(child);
            assert(child_stored == child);
#endif
            while (child->layer < node->layer - 1)
                child = add_node(make_dummy(child));
        }
        return add_node(node);
    }

    /**
     * Remove all nodes from this circuit that are not reachable from any root.
     *
     * Non-input, non-root nodes that are not children of any reachable node are removed.
     */
    void remove_unused_nodes() {
        if (nb_layers() == 1)
            return;

        std::vector<std::vector<bool>> used(nb_layers());
        for (std::size_t i = 1; i < nb_layers(); ++i)
            used[i].resize(layers[i].size(), false);

        for (auto& root : roots) {
            if (root->layer != 0)
                used[root->layer][root->ix] = true;
        }

        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            for (auto& node : *it) {
                if (node->layer == 0)
                    continue;
                assert(node->layer < used.size());
                assert(node->ix < used[node->layer].size());
                if (used[node->layer][node->ix]) {
                    for (auto child : node->children) {
                        if (child->layer == 0)
                            continue;
                        used[child->layer][child->ix] = true;
                    }
                }
            }
        }

        for (std::size_t i = 1; i < nb_layers(); ++i) {
            for (auto it = layers[i].begin(); it != layers[i].end();) {
                if (!used[i][(*it)->ix]) {
                    Node* del_node = *it;
                    it = layers[i].erase(it);
                    delete del_node;
                } else {
                    ++it;
                }
            }
        }

        // Pop empty trailing layers (intermediate layers can't be empty due to dummy nodes)
        for (std::size_t i = nb_layers() - 1; i > 0; --i) {
            if (layers[i].empty()) {
                layers.pop_back();
                layer_gate_types.pop_back();
            } else {
                break;
            }
        }

        // Update ix to be contiguous [0..n-1] within each layer
        for (std::size_t i = 1; i < nb_layers(); ++i) {
            assert(!layers[i].empty());
            int index = 0;
            for (auto& node : layers[i])
                node->ix = index++;
        }

#ifndef NDEBUG
        if (layers.size() > 2) {
            for (std::size_t i = 2; i < nb_layers(); ++i) {
                for (auto& node : layers[i]) {
                    for (auto& child : node->children) {
                        assert(child->ix < layers[i - 1].size());
                    }
                }
            }
        }
#endif
    }

    /**
     * Resolve the layer for a gate of the given type, starting from min_layer.
     * Scans existing layers for a match; if none found, appends a new layer.
     */
    std::size_t resolve_layer(int gate_type, std::size_t min_layer) {
        for (std::size_t l = min_layer; l < layer_gate_types.size(); ++l) {
            if (layer_gate_types[l] == gate_type)
                return l;
        }
        std::size_t new_layer = layer_gate_types.size();
        layer_gate_types.push_back(gate_type);
        return new_layer;
    }

    /**
     * Move all roots into a new top layer, inserting dummy chains as needed.
     * The final layer uses root order as the hash (so CSR output matches root order).
     */
    void add_root_layer() {
        if (roots.empty())
            throw std::runtime_error("Cannot construct root layer, there are no roots!");

        std::size_t root_layer_index = nb_layers();
        for (std::size_t i = 0; i < roots.size(); i++) {
            Node* root = roots[i];
            while (root->layer < root_layer_index) {
                Node* dummy = make_dummy(root);
                if (dummy->layer == root_layer_index)
                    dummy->hash = i; // order final-layer nodes by root insertion order
                root = add_node(dummy);
            }
            roots[i] = root;
        }
    }

protected:
    /**
     * Create a passthrough (dummy) Gate node one layer above `child`.
     */
    static Node* make_dummy(Node* child) {
        Node* dummy = Node::createInternal(-1);
        dummy->add_child(child);
        return dummy;
    }

    /**
     * Insert `node` into its layer's hash set; deduplicates automatically.
     *
     * Takes ownership of `node`. If an equivalent node already exists, `node` is
     * freed (deleted) and the existing node is returned. Otherwise `node` itself
     * is returned (and `node->ix` is assigned).
     */
    Node* add_node(Node* node) {
        if (layers.size() <= node->layer) {
            layers.resize(node->layer + 1);
            layer_gate_types.resize(layers.size(), -1);
        }
        auto& layer = layers[node->layer];
        auto [it, inserted] = layer.insert(node);
        if (inserted && node->ix == -1)
            node->ix = layer.size() - 1;
        if (*it != node)
            delete node;
        return *it;
    }
};
