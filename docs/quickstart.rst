Quick Start Guide
=================


Installation
************

KLay supports Linux, Mac and Windows. Make sure you have installed Python, and install KLay using pip.

>>> pip install klaycircuits

To build the latest version of KLay from source, download the repo and run:

>>> pip install .


Usage
*****

First, we need to create a circuit. You can both manually define the circuit, or import it from a knowledge compiler.
For more information, check out the :ref:`circuit_construction`.

.. code-block:: Python

   from klaycircuits import Circuit

   circuit = Circuit()
   circuit.add_sdd(sdd_node)

Now that we have the circuit, we can evaluate it. To do this, we first turn the circuit into a PyTorch module.

.. code-block:: Python

   import torch

   module = circuit.to_torch_module()
   module = module.to("cuda:0")

We can use our circuit as any other PyTorch module.
The input should be a tensor with the weights for each literal, and the output is the result of evaluating circuit.
For more details, see the :ref:`circuit_eval`.

.. code-block:: Python

   weights = torch.tensor([...], device="cuda:0")
   result = module(weights)
   result.backward()

