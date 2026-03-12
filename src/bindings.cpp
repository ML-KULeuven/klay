#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <sstream>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>

#include "logical_circuit.h"

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// NodePtr: binding-layer handle for a Node* owned by a LogicalCircuit.
//
// A value-type wrapper so nanobind has a named, copyable Python type without
// any ownership or deletion semantics — the Circuit owns the actual nodes.
// ---------------------------------------------------------------------------

class NodePtr {
public:
    explicit NodePtr(Node* ptr) : ptr(ptr) {}

    Node* get() const { return ptr; }

    std::string to_string() const {
        std::ostringstream ss;
        ss << "NodePtr(" << as_int() << ")";
        return ss.str();
    }

    bool operator==(NodePtr other) const { return ptr == other.ptr; }

    std::uintptr_t as_int() const { return reinterpret_cast<std::uintptr_t>(ptr); }

private:
    Node* ptr;
};

// ---------------------------------------------------------------------------
// numpy array conversion for _get_indices
// ---------------------------------------------------------------------------

using NdArray = nb::ndarray<nb::numpy, long, nb::shape<-1>>;
using Arrays  = std::vector<NdArray>;

static auto get_indices_as_arrays(LogicalCircuit& c) {
    auto [indices_vecs, csr_vecs, layer_types] = c.get_indices();

    auto to_ndarray = [](std::vector<long> vec) -> NdArray {
        long* data = new long[vec.size()];
        std::copy(vec.begin(), vec.end(), data);
        std::size_t shape[1] = {vec.size()};
        nb::capsule cap(data, [](void* p) noexcept { delete[] static_cast<long*>(p); });
        return NdArray(data, 1, shape, cap);
    };

    Arrays indices_ndarrays, csr_ndarrays;
    for (auto& v : indices_vecs) indices_ndarrays.push_back(to_ndarray(std::move(v)));
    for (auto& v : csr_vecs)     csr_ndarrays.push_back(to_ndarray(std::move(v)));

    return std::make_tuple(indices_ndarrays, csr_ndarrays, layer_types);
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

NB_MODULE(klay_ext, m) {
m.doc() = "Layerize arithmetic circuits";

nb::class_<NodePtr>(m, "NodePtr")
.def("__repr__", &NodePtr::to_string)
.def(nb::self == nb::self)
.def("__hash__", &NodePtr::as_int)
.def("get_ix", [](NodePtr a) { return a.get()->ix; });

nb::class_<LogicalCircuit>(m, "Circuit",
    "Circuits are the main class added by KLay, and require no arguments to construct.\n\n"
    ":code:`circuit = klay.Circuit()` ")
.def(nb::init<>())
.def("add_sdd_from_file",
    [](LogicalCircuit& c, const std::string& f, std::vector<int> tl, std::vector<int> fl) {
        return NodePtr(c.add_sdd_from_file(f, tl, fl));
    },
    "filename"_a, "true_lits"_a = std::vector<int>(), "false_lits"_a = std::vector<int>(),
    "Add a sentential decision diagram (SDD) from file.\n\n"
    ":param filename:\n\tPath to the :code:`.sdd` file on disk.\n"
    ":param true_lits:\n\tList of literals that are always true and should get propagated away.\n"
    ":param false_lits:\n\tList of literals that are always false and should get propagated away.")
.def("add_d4_from_file",
    [](LogicalCircuit& c, const std::string& f, std::vector<int> tl, std::vector<int> fl) {
        return NodePtr(c.add_d4_from_file(f, tl, fl));
    },
    "filename"_a, "true_lits"_a = std::vector<int>(), "false_lits"_a = std::vector<int>(),
    "Add an NNF circuit in the D4 format from file.\n\n"
    ":param filename:\n\tPath to the :code:`.nnf` file on disk.\n"
    ":param true_lits:\n\tList of literals that are always true and should get propagated away.\n"
    ":param false_lits:\n\tList of literals that are always false and should get propagated away.")
.def("_get_indices", &get_indices_as_arrays)
.def("nb_nodes", &LogicalCircuit::nb_nodes, "Number of nodes in the circuit.")
.def("nb_root_nodes", &LogicalCircuit::nb_root_nodes, "Number of root nodes in the circuit.")
.def("true_node",    [](LogicalCircuit& c) { return NodePtr(c.true_node()); },
    "Adds a true node to the circuit, and returns a pointer to this node.")
.def("false_node",   [](LogicalCircuit& c) { return NodePtr(c.false_node()); },
    "Adds a false node to the circuit, and returns a pointer to this node.")
.def("literal_node", [](LogicalCircuit& c, int lit) { return NodePtr(c.literal_node(lit)); },
    "Adds a literal node to the circuit, and returns a pointer to this node.", "literal"_a)
.def("or_node",  [](LogicalCircuit& c, std::vector<NodePtr> children) {
        std::vector<Node*> raw;
        for (auto& np : children) raw.push_back(np.get());
        return NodePtr(c.or_node(raw));
    }, "children"_a, "Adds an :code:`or` node to the circuit, and returns a pointer to this node.")
.def("and_node", [](LogicalCircuit& c, std::vector<NodePtr> children) {
        std::vector<Node*> raw;
        for (auto& np : children) raw.push_back(np.get());
        return NodePtr(c.and_node(raw));
    }, "children"_a, "Adds an :code:`and` node to the circuit, and returns a pointer to this node.")
.def("set_root", [](LogicalCircuit& c, NodePtr root) { c.set_root(root.get()); }, "root"_a,
    "Marks a node pointer as root. The order in which nodes are set as root determines the order of the output tensor.\n"
    " .. note:: Only use this when manually constructing a circuit, when loading in a NNF/SDD its root is automatically set as root.\n")
.def("remove_unused_nodes", &LogicalCircuit::remove_unused_nodes,
    "Removes unused nodes from the circuit. Root nodes are always considered used.\n"
    " .. warning:: Invalidates any :code:`NodePtr` referring to an unused node "
    "(i.e., a node not connected to a root node).\n");

m.def("to_dot_file", &to_dot_file, "circuit"_a, "filename"_a,
    "Write the given circuit as dot format to a file");
}
