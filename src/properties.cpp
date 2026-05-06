#include "klay/properties.h"
#include "klay/util.h"

namespace klay {

std::string sdnnf_summary(const SDNNFResult& r) {
    auto tick = [](bool b) -> const char* { return b ? "v" : "x"; };
    std::string s;
    s += "\n";
    s += "  NNF            "; s += tick(r.is_nnf); s += "     (klay guarantee)\n";
    s += "  Decomposable   "; s += tick(r.is_decomposable); s += "\n";
    s += "  Smooth         "; s += tick(r.is_smooth);       s += "\n";
    s += "  ----------------------------------------\n";
    s += "  DNNF           "; s += tick(r.is_dnnf());       s += "\n";
    s += "  s-DNNF         "; s += tick(r.is_sdnnf());      s += "\n";
    s += "  ----------------------------------------\n";
    s += "  AND nodes: ";     s += std::to_string(r.n_and); s += "\n";
    s += "  OR  nodes: ";     s += std::to_string(r.n_or);  s += "\n";
    s += "  ----------------------------------------\n";
    if (!r.violations.empty()) {
        s += "  Violations (" + std::to_string(r.violations.size()) + "):\n";
        for (const auto& v : r.violations)
            s += "    * [" + v.property + "] node_ix=" +
                 std::to_string(v.ix) + ": " + v.detail + "\n";
    }
    return s;
}

static Support
compute_support(const Node* node,
                const SupportMap& support_of,
                std::size_t n_words) {
    Support s(n_words, 0);

    switch (node->type) {
        case NodeType::True:
        case NodeType::False:
            break;

        case NodeType::Leaf: {
            std::size_t var = static_cast<std::size_t>(node->ix) >> 1;
            support_var(s, var);
            break;
        }

        case NodeType::And:
        case NodeType::Or:
            for (const auto* child : node->children) {
                assert(support_of.count(child) > 0);
                support_union(s, support_of.at(child));
            }
            break;

        default:
            break;
    }

    return s;
}

// ---------------------------------------------------------------------------
// An AND node is decomposable if no variable appears in more than one child.
// (https://arxiv.org/pdf/cs/0003044)
// ---------------------------------------------------------------------------
class DecomposabilityChecker final : public IPropertyChecker {
 public:
    void on_node(const Node* node,
               const SupportMap& support_of,
               std::size_t max_violations,
               SDNNFResult& result) override {
        if (node->type != NodeType::And) return;
        if (node->children.size() <= 1) return;  // dummy node

        assert(support_of.count(node) >= 1);
        const std::size_t nw = support_of.at(node).size();
        Support running(nw, 0);

        bool violated = false;
        int  child_k  = 0;

        for (const auto* child : node->children) {
            const Support& cs = support_of.at(child);

            if (!violated && support_intersect(running, cs)) {
                result.is_decomposable = false;
                violated = true;
                if (result.violations.size() < max_violations) {
                    std::ostringstream detail;
                    detail << "child " << child_k
                           << " (support="
                           << support_to_string(cs, result.n_vars_found)
                           << ") overlaps other children's support="
                           << support_to_string(running, result.n_vars_found);
                    result.violations.push_back({
                        "decomposability", node->ix,
                        node->layer, node->hash, detail.str()
                    });
                }
            }

            support_union(running, cs);
            ++child_k;
        }
    }
};

// ---------------------------------------------------------------------------
// An OR node is smooth if every child mentions exactly
// the same set of variables.
// (https://arxiv.org/pdf/cs/0003044)
// ---------------------------------------------------------------------------
class SmoothnessChecker final : public IPropertyChecker {
 public:
    void on_node(const Node* node,
                 const SupportMap& support_of,
                 std::size_t max_violations,
                 SDNNFResult& result) override {
        if (node->type != NodeType::Or)  return;
        if (node->children.size() <= 1)  return;  // single-child dummy node

        const Support& ref = support_of.at(node->children.front());

        bool violated = false;
        int  child_k  = 1;

        for (auto it = std::next(node->children.begin());
             it != node->children.end(); ++it, ++child_k) {
            const Support& cs = support_of.at(*it);

            if (!violated && !support_equal(ref, cs)) {
                result.is_smooth = false;
                violated = true;
                if (result.violations.size() < max_violations) {
                    std::ostringstream detail;
                    auto n_vars = result.n_vars_found;
                    detail << "child 0 scope="
                      << support_to_string(ref, n_vars)
                            << ", child "
                            << child_k
                            << " scope="
                            << support_to_string(cs, n_vars)
                            << ", symmetric difference="
                            << support_sym_diff_string(ref, cs, n_vars);
                    result.violations.push_back({
                        "smoothness", node->ix,
                        node->layer, node->hash, detail.str()
                    });
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// An OR node is deterministic if its children are disjoint.
// (https://arxiv.org/pdf/cs/0003044)
//
// Checking determinism of an arbitrary NNF/arithmetic circuit is coNP-complete,
// (SAT on branch_i ∧ branch_j is NP-complete)
// However, if it is known that the circuit is decomposable, one might have a
// more efficient check. Or if it is known that the circuit is not smooth, one
// might have it easier.
//
// A not complete check might be to check if there
// is literal present in one branch that is the complement
// in another etc.
// ---------------------------------------------------------------------------

std::unique_ptr<IPropertyChecker> make_decomposability_checker() {
    return std::make_unique<DecomposabilityChecker>();
}

std::unique_ptr<IPropertyChecker> make_smoothness_checker() {
    return std::make_unique<SmoothnessChecker>();
}

SDNNFResult run_checks(const Circuit& circuit,
                       std::size_t max_violations,
                       std::vector<std::unique_ptr<IPropertyChecker>> checkers) {
    SDNNFResult result;

    if (circuit.nb_layers() == 0)
        return result;

    // (1) detect variables
    std::set<std::size_t> vars_found;
    for (const auto* node : circuit.layers[0]) {
        if (node->type == NodeType::Leaf) {
            std::size_t var = static_cast<std::size_t>(node->ix) >> 1;
            vars_found.insert(var);
        }
    }

    result.n_vars_found = vars_found.size();
    if (result.n_vars_found == 0)
        return result;

    const std::size_t nw = n_words(result.n_vars_found);

    // (2) bottom-up traversal
    SupportMap support_of;
    support_of.reserve(circuit.nb_nodes());

    for (const auto& layer : circuit.layers) {
        for (const auto* node : layer) {
            // (a) compute and store this node's support
            support_of[node] = compute_support(node, support_of, nw);

            // (b) count node types
            if (node->type == NodeType::And) ++result.n_and;
            if (node->type == NodeType::Or)  ++result.n_or;

            // (c) run all registered checkers
            for (auto& checker : checkers)
                checker->on_node(node, support_of, max_violations, result);
        }
    }
    // (d) run registered checkers at end
    for (auto& checker : checkers)
        checker->finalize(max_violations, result);

    return result;
}

SDNNFResult check_sdnnf(const Circuit& circuit,
                        std::size_t max_violations) {
    std::vector<std::unique_ptr<IPropertyChecker>> checkers;
    checkers.push_back(make_decomposability_checker());
    checkers.push_back(make_smoothness_checker());
    return run_checks(circuit, max_violations, std::move(checkers));
}

SDNNFResult check_decomposability(const Circuit& circuit,
                                  std::size_t max_violations) {
    std::vector<std::unique_ptr<IPropertyChecker>> checkers;
    checkers.push_back(make_decomposability_checker());
    return run_checks(circuit, max_violations, std::move(checkers));
}

SDNNFResult check_smooth(const Circuit& circuit,
                         std::size_t max_violations) {
    std::vector<std::unique_ptr<IPropertyChecker>> checkers;
    checkers.push_back(make_smoothness_checker());
    return run_checks(circuit, max_violations, std::move(checkers));
}

}  // namespace klay
