.. _circuit_construction:

Circuit Creation
================

There are several ways to create a circuit. You can either load in a circuit compiled by a PySDD or d4, or you can manually the circuit.

Loading Circuits
********************

An SDD can be loaded from a file as follows.

.. code-block:: Python

   from klaycircuits import Circuit

   circuit = Circuit()
   circuit.add_sdd_from_file("path/to/my.sdd")

Similarly, for d4 we can use

.. code-block:: Python

   circuit = Circuit()
   circuit.add_d4_from_file("path/to/my.nnf")

SDDs can also be loaded directly from a PySDD :code:`SddNode` object.

.. code-block:: Python

   from pysdd.sdd import SddManager

   manager = SddManager(var_count = 2)
   sdd_node = manager.literal(1) & manager.literal(2)

   circuit = Circuit()
   circuit.add_sdd(sdd_node)


Multi-Rooted Circuits
*********************

If you want to evaluate multiple circuits in parallel, you can merge them into a single multi-rooted circuit.

.. code-block:: Python

   circuit = Circuit()
   circuit.add_sdd(first_sdd)
   circuit.add_sdd(second_sdd)

Evaluating this circuit will result in an output tensor with two elements. The order in which the circuits are added
determines the order in the output when evaluating.


Manual Circuits
***************************

If you want to create a custom circuit, you can manually define the circuit structure.
We start by defining some literals.

.. code-block:: Python

    circuit = Circuit()
    a = circuit.literal_node(1)
    b = circuit.literal_node(-2)

Next, create and/or nodes as follows.

.. code-block:: Python

    and_node = circuit.and_node([a, b])

To indicate that a node is a root (i.e. it will be part of the output), you need to mark it as root.

.. code-block:: Python

    circuit.set_root(and_node)

As we support multi-rooted circuits, you can later add more nodes and mark them as root.

.. code-block:: Python

    or_node = circuit.or_node([a, b])
    circuit.set_root(or_node)

