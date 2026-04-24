#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>

#include "klay/circuit.h"
#include "klay/indices.h"
#include "klay/properties.h"

namespace klay {

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(klay_ext, m) {

nb::class_<NodePtr>(m, "NodePtr")
.def("__repr__", &NodePtr::to_string)
.def(nb::self == nb::self)
.def("__hash__", &NodePtr::as_int)
.def("get_ix", [](NodePtr a) {return a.get()->ix;});

nb::class_<Circuit>(m, "Circuit", "Circuits are the main class added by KLay, and require no arguments to construct.\n\n:code:`circuit = klay.Circuit()` ")
.def(nb::init<>())
.def("add_sdd_from_file", &Circuit::add_sdd_from_file, "filename"_a, "true_lits"_a = std::vector<int>(), "false_lits"_a = std::vector<int>(), "Add a sentential decision diagram (SDD) from file.\n\n:param filename:\n\tPath to the :code:`.sdd` file on disk.\n:param true_lits:\n\tList of literals that are always true and should get propagated away.\n:param false_lits:\n\tList of literals that are always false and should get propagated away.")
.def("add_d4_from_file", &Circuit::add_d4_from_file, "filename"_a, "true_lits"_a = std::vector<int>(), "false_lits"_a = std::vector<int>(), "Add an NNF circuit in the D4 format from file.\n\n:param filename:\n\tPath to the :code:`.nnf` file on disk.\n:param true_lits:\n\tList of literals that are always true and should get propagated away.\n:param false_lits:\n\tList of literals that are always false and should get propagated away.")
.def("nb_nodes", &Circuit::nb_nodes, "Number of nodes in the circuit.")
.def("nb_root_nodes", &Circuit::nb_root_nodes, "Number of root nodes in the circuit.")
.def("true_node", &Circuit::true_node, "Adds a true node to the circuit, and returns a pointer to this node.")
.def("false_node", &Circuit::false_node, "Adds a false node to the circuit, and returns a pointer to this node.")
.def("literal_node", &Circuit::literal_node, "Adds a literal node to the circuit, and returns a pointer to this node.", "literal"_a)
.def("or_node", &Circuit::or_node, "children"_a, "Adds an :code:`or` node to the circuit, and returns a pointer to this node.")
.def("and_node", &Circuit::and_node, "children"_a, "Adds an :code:`and` node to the circuit, and returns a pointer to this node.")
.def("set_root", &Circuit::set_root, "root"_a, "Marks a node pointer as root. The order in which nodes are set as root determines the order of the output tensor.\n .. note:: Only use this when manually constructing a circuit, when loading in a NNF/SDD its root is automatically set as root.\n")
.def("remove_unused_nodes", &Circuit::remove_unused_nodes, "Removes unused nodes from the circuit. Root nodes are always considered used.\n .. warning:: Invalidates any :code:`NodePtr` referring to an unused node (i.e., a node not connected to a root node).\n")
.def("print", &Circuit::print_circuit, "Print the circuit structure to stdout.")
.def("_get_indices", &get_indices);

nb::class_<SDNNFViolation>(m, "SDNNFViolation")
    .def_ro("property",  &SDNNFViolation::property)
    .def_ro("node_hash", &SDNNFViolation::node_hash)
    .def_ro("ix",        &SDNNFViolation::ix)
    .def_ro("layer",     &SDNNFViolation::layer)
    .def_ro("detail",    &SDNNFViolation::detail)
    .def("__repr__", [](const SDNNFViolation& v) {
        return "[" + v.property + "] node_hash=" +
               std::to_string(v.node_hash) + ": " + v.detail;
    });

nb::class_<SDNNFResult>(m, "SDNNFResult")
    .def_ro("is_nnf",             &SDNNFResult::is_nnf)
    .def_ro("is_decomposable",    &SDNNFResult::is_decomposable)
    .def_ro("is_smooth",          &SDNNFResult::is_smooth)
    .def_ro("n_and",              &SDNNFResult::n_and)
    .def_ro("n_or",               &SDNNFResult::n_or)
    .def_ro("n_vars_found",       &SDNNFResult::n_vars_found)
    .def_ro("violations",         &SDNNFResult::violations)
    .def_prop_ro("is_dnnf",       &SDNNFResult::is_dnnf)
    .def_prop_ro("is_sdnnf",      &SDNNFResult::is_sdnnf)
    .def("summary",               &sdnnf_summary);

m.def("check_sdnnf", &check_sdnnf,
      "circuit"_a,
      "max_violations"_a = std::size_t(50),
      "check whether a klay circuit satisfies (smooth) d-DNNF. "
      "returns an SDNNFResult.");

m.def("check_decomposability", &check_decomposability,
      "circuit"_a,
      "max_violations"_a = std::size_t(50),
      "check whether a klay circuit satisfies decomposability. "
      "returns an SDNNFResult.");

m.def("check_smooth", &check_smooth,
      "circuit"_a,
      "max_violations"_a = std::size_t(50),
      "check whether a klay circuit satisfies smoothness. "
      "returns an SDNNFResult.");
}

}  // namespace klay
