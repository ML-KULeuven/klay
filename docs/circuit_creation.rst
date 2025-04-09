.. _circuit_construction:

Circuit Creation Tutorial
=========================

There are two ways to create a circuit. You can either manually specify the circuit or you can load in circuit from a knowledge compiler (currently PySDD and d4 are supported).

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

To evaluate multiple circuits in parallel, you can merge them into a single circuit with multiple roots.

.. code-block:: Python

   circuit = Circuit()
   circuit.add_sdd(first_sdd)
   circuit.add_sdd(second_sdd)

Evaluating this circuit will result in an output tensor with two elements. The order in which the circuits are added
determines the order of the roots in the output tensor.


Manual Circuits
***************************

To create a custom circuit, you can manually define the circuit structure.
We start by defining some literal nodes, which are the leafs of the circuit.

.. code-block:: Python

    circuit = Circuit()
    a = circuit.literal_node(1)
    b = circuit.literal_node(-2)

Next, create `and`/`or` nodes as follows.

.. code-block:: Python

    and_node = circuit.and_node([a, b])
    or_node = circuit.or_node([a, and_node])

You need to set a node as root to indicate that it will be part of the output.

.. code-block:: Python

    circuit.set_root(or_node)

As we support multi-rooted circuits, you can later add other root nodes.

.. code-block:: Python

    circuit.set_root(and_node)

