#include "logical_circuit.h"

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
.def("add_sdd_from_file", &LogicalCircuit::add_sdd_from_file,
    "filename"_a, "true_lits"_a = std::vector<int>(), "false_lits"_a = std::vector<int>(),
    "Add a sentential decision diagram (SDD) from file.\n\n"
    ":param filename:\n\tPath to the :code:`.sdd` file on disk.\n"
    ":param true_lits:\n\tList of literals that are always true and should get propagated away.\n"
    ":param false_lits:\n\tList of literals that are always false and should get propagated away.")
.def("add_d4_from_file", &LogicalCircuit::add_d4_from_file,
    "filename"_a, "true_lits"_a = std::vector<int>(), "false_lits"_a = std::vector<int>(),
    "Add an NNF circuit in the D4 format from file.\n\n"
    ":param filename:\n\tPath to the :code:`.nnf` file on disk.\n"
    ":param true_lits:\n\tList of literals that are always true and should get propagated away.\n"
    ":param false_lits:\n\tList of literals that are always false and should get propagated away.")
.def("_get_indices", &LogicalCircuit::get_indices)
.def("nb_nodes", &LogicalCircuit::nb_nodes, "Number of nodes in the circuit.")
.def("nb_root_nodes", &LogicalCircuit::nb_root_nodes, "Number of root nodes in the circuit.")
.def("true_node",    &LogicalCircuit::true_node,    "Adds a true node to the circuit, and returns a pointer to this node.")
.def("false_node",   &LogicalCircuit::false_node,   "Adds a false node to the circuit, and returns a pointer to this node.")
.def("literal_node", &LogicalCircuit::literal_node, "Adds a literal node to the circuit, and returns a pointer to this node.", "literal"_a)
.def("or_node",      &LogicalCircuit::or_node,  "children"_a, "Adds an :code:`or` node to the circuit, and returns a pointer to this node.")
.def("and_node",     &LogicalCircuit::and_node, "children"_a, "Adds an :code:`and` node to the circuit, and returns a pointer to this node.")
.def("set_root",     &LogicalCircuit::set_root, "root"_a,
    "Marks a node pointer as root. The order in which nodes are set as root determines the order of the output tensor.\n"
    " .. note:: Only use this when manually constructing a circuit, when loading in a NNF/SDD its root is automatically set as root.\n")
.def("remove_unused_nodes", &LogicalCircuit::remove_unused_nodes,
    "Removes unused nodes from the circuit. Root nodes are always considered used.\n"
    " .. warning:: Invalidates any :code:`NodePtr` referring to an unused node "
    "(i.e., a node not connected to a root node).\n");

m.def("to_dot_file", &to_dot_file, "circuit"_a, "filename"_a,
    "Write the given circuit as dot format to a file");
}
