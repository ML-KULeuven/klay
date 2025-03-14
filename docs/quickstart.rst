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
So far KLay supports the PySDD and d4 circuit formats.
For example, you can load in a PySDD circuit as follows.

.. code-block:: Python

   import klaycircuits

   circuit = klaycircuits.Circuit()
   circuit.add_sdd(sdd_node)

For more information on circuit construction, check out <todo>.

Now that we have the circuit, we can evaluate it. To do this, we first turn the circuit into a PyTorch module.
Other backends such as Jax are also supported.

.. code-block:: Python

   import torch

   module = circuit.to_torch_module()

We can use this module as any other PyTorch module. The expected input is a tensor with the weights for each literal.

.. code-block:: Python

   weights = torch.tensor([...])
   result = module(weights)
   result.backward()