#pragma once

#include "node.h"

/**
 * Layer policy for a standard AND/OR circuit.
 *
 * Invariant: And nodes live in odd layers, Or nodes in even layers.
 *
 * This policy is used as a template parameter for StratifiedDag.
 * It must provide a single static method:
 *   static Node* make_dummy(Node* child)
 *
 * The Policy concept requires:
 *   static Node* make_dummy(Node* child)
 *     Create a passthrough (dummy) node one layer above `child`.
 *     The returned node has `child` as its only child and is not yet
 *     inserted into the circuit.
 */
struct AndOrPolicy {
    /**
     * Create a dummy passthrough node one layer above `child`.
     *
     * If `child` is at an even layer, the dummy is an And node (odd layer).
     * If `child` is at an odd layer, the dummy is an Or node (even layer).
     */
    static Node* make_dummy(Node* child) {
        Node* dummy = (child->layer % 2 == 0) ? Node::createAndNode() : Node::createOrNode();
        dummy->add_child(child);
        // Parity is naturally satisfied: even child -> layer+1 is odd -> And; odd child -> Or.
        return dummy;
    }
};
