import os.path

from graphviz import Source
from pysdd.sdd import SddManager, Vtree

def view_sdd(filepath_sdd=None, filepath_vtree=None):
    if filepath_sdd is None:
        filepath_sdd = "test.sdd"
    if filepath_vtree is None:
        filepath_vtree = "test.vtree"

    # check that filepath_sdd exists
    assert os.path.exists(filepath_sdd), f"File {filepath_sdd} does not exist"
    assert os.path.exists(filepath_sdd), f"File {filepath_vtree} does not exist"

    vtree = Vtree.from_file(filepath_vtree.encode())
    manager = SddManager.from_vtree(vtree)
    formula = manager.read_sdd_file(filepath_sdd.encode())
    Source(formula.dot()).render("sdd_plot", format="pdf", cleanup=True, view=True)

def view_dot_file(filepath_dot=None):
    if filepath_dot is None:
        filepath_dot = "circuit.dot"
    assert os.path.exists(filepath_dot), f"File {filepath_dot} does not exist"
    Source.from_file(filepath_dot).render("circuit_plot", format="pdf", cleanup=True, view=True)


if __name__ == "__main__":
    view_dot_file()

