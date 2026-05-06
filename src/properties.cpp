#include <string>
#include <set>

#include "klay/properties.h"
#include "klay/util.h"

namespace klay {

std::string sdnnf_summary(const SDNNFResult& r) {
    auto tick = [](bool b) -> const char* { return b ? "v" : "x"; };
    std::string s;
    s += "\n";
    s += "  NNF            "; s += tick(r.is_nnf); s += "   (klay guarantee)\n";
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


static SupportMap
build_support_map(const Circuit& circuit, SDNNFResult& result) {
    std::set<std::size_t> vars_found;
    for (const auto* node : circuit.layers[0])
        if (node->type == NodeType::Leaf)
            vars_found.insert(static_cast<std::size_t>(node->ix) >> 1);

    result.n_vars_found = vars_found.size();
    if (result.n_vars_found == 0) return {};

    const std::size_t nw = n_words(result.n_vars_found);
    SupportMap support_of;
    support_of.reserve(circuit.nb_nodes());

    for (const auto& layer : circuit.layers)
        for (const auto* node : layer) {
            support_of[node] = compute_support(node, support_of, nw);
            if (node->type == NodeType::And) ++result.n_and;
            if (node->type == NodeType::Or)  ++result.n_or;
        }

    return support_of;
}


static void
check_or_smooth(const Node* node,
                const SupportMap& support_of,
                std::size_t max_violations,
                SDNNFResult& result) {
    const Support& ref = support_of.at(node->children.front());
    int child_k = 1;

    for (const auto* child : node->children) {
        const Support& cs = support_of.at(child);
        if (!support_equal(ref, cs)) {
            result.is_smooth = false;
            if (result.violations.size() < max_violations) {
                std::ostringstream detail;
                detail << "child 0 scope="
                       << support_to_string(ref, result.n_vars_found)
                       << ", child " << child_k
                       << " scope="
                       << support_to_string(cs, result.n_vars_found)
                       << ", symmetric difference="
                       << support_sym_diff_string(ref, cs, result.n_vars_found);
                result.violations.push_back(
                    {"smoothness",
                      node->ix,
                      node->layer,
                      node->hash,
                      detail.str()});
            }
            return;
        }
    }
}


static void
check_and_decomp(const Node* node,
                 const SupportMap& support_of,
                 std::size_t max_violations,
                 SDNNFResult& result) {
    const std::size_t nw = support_of.at(node).size();
    Support running(nw, 0);
    int child_k = 0;

    for (const auto* child : node->children) {
        const Support& cs = support_of.at(child);
        if (support_intersect(running, cs)) {
            result.is_decomposable = false;
            if (result.violations.size() < max_violations) {
                std::ostringstream detail;
                detail << "child " << child_k
                       << " (support="
                       << support_to_string(cs, result.n_vars_found)
                       << ") overlaps running support="
                       << support_to_string(running, result.n_vars_found);
                result.violations.push_back(
                    {"decomposability",
                      node->ix,
                      node->layer,
                      node->hash,
                      detail.str()});
            }
            return;
        }
        support_union(running, cs);
        ++child_k;
    }
}


// ---------------------------------------------------------------------------
// An AND node is decomposable if no variable appears in more than one child.
// (https://arxiv.org/pdf/cs/0003044)
// ---------------------------------------------------------------------------
SDNNFResult
check_decomposability(const Circuit& circuit, std::size_t max_violations) {
    SDNNFResult result;
    if (circuit.nb_layers() == 0) return result;

    SupportMap support_of = build_support_map(circuit, result);
    if (result.n_vars_found == 0) return result;

    for (const auto& layer : circuit.layers) {
        for (const auto* node : layer) {
            if (node->type != NodeType::And ||
                node->children.size() <= 1)
              continue;
            check_and_decomp(node, support_of, max_violations, result);
        }
    }
    return result;
}


// ---------------------------------------------------------------------------
// An OR node is smooth if every child mentions exactly
// the same set of variables.
// (https://arxiv.org/pdf/cs/0003044)
// ---------------------------------------------------------------------------
SDNNFResult
check_smooth(const Circuit& circuit, std::size_t max_violations) {
    SDNNFResult result;
    if (circuit.nb_layers() == 0) return result;

    SupportMap support_of = build_support_map(circuit, result);
    if (result.n_vars_found == 0) return result;

    for (const auto& layer : circuit.layers) {
        for (const auto* node : layer) {
            if (node->type != NodeType::Or ||
                node->children.size() <= 1) continue;
            check_or_smooth(node, support_of, max_violations, result);
        }
    }
    return result;
}


SDNNFResult check_sdnnf(const Circuit& circuit, std::size_t max_violations) {
    SDNNFResult result;
    if (circuit.nb_layers() == 0) return result;

    SupportMap support_of = build_support_map(circuit, result);
    if (result.n_vars_found == 0) return result;

    for (const auto& layer : circuit.layers)
        for (const auto* node : layer) {
            if (node->type == NodeType::And && node->children.size() > 1)
                check_and_decomp(node, support_of, max_violations, result);
            if (node->type == NodeType::Or  && node->children.size() > 1)
                check_or_smooth(node, support_of, max_violations, result);
        }

    return result;
}

}  // namespace klay
