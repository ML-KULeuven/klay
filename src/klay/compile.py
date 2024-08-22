import platform
from functools import reduce
from pathlib import Path

from pysdd.sdd import SddManager, Vtree, SddNode


def compile_sdd(dimacs_file: str) -> SddNode:
    """
    Compile a CNF formula from a DIMACS file into an SDD.

    Args:
        dimacs_file: Path to the DIMACS file.

    Returns:
        SddNode: The compiled Circuit.
    """
    with open(dimacs_file) as f:
        header = f.readline().split(" ")
        assert header[0] == "p" and header[1] == "cnf"
        var_count = int(header[2])
        clause_count = int(header[3])

        vtree = Vtree(var_count=var_count, vtree_type="balanced")
        manager = SddManager.from_vtree(vtree)
        sdd = manager.true()

        for line in f:
            clause = (int(x) for x in line.strip().split(" "))
            literals = (manager.l(x) for x in clause if x != 0)
            sdd &= reduce(lambda x, y: x | y, literals)

    return sdd


def _get_d4_path() -> str:
    """ Get the path to the d4 binary for the current system. """
    lib_path = Path(__file__).parent / "lib"
    system = f"{platform.system()}-{platform.processor()}"
    d4_path = lib_path / system / "d4"
    assert d4_path.exists(), f"Could not find d4 for your system {system} in {d4_path}"
    d4_path.chmod(0o755)  # Set binary as executable
    return str(d4_path)


def compile_d4(dimacs_file: str, nnf_file: str):
    """
    Compile a CNF formula from a DIMACS file into a d-DNNF using d4.

    Args:
        dimacs_file: Path to the dimacs file.
        nnf_file: Path to store the compiled circuit in NNF.
    """
    d4_path = _get_d4_path()
    import subprocess
    result = subprocess.run(
        [d4_path, "-dDNNF", dimacs_file, f"-out={nnf_file}"],
        stdout=subprocess.DEVNULL
    )
    assert result.returncode == 0, f"Failed to compile {dimacs_file} to {nnf_file}"
