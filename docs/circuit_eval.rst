.. _circuit_eval:

Circuit Evaluation Tutorial
===========================

Backends
********

Once we have created a circuit, we can start using it. KLay relies on a backend to perform the inference. Currently, the PyTorch and Jax backends are implemented.

.. tabs::

    .. group-tab:: PyTorch

        We can turn the circuit into a PyTorch module as follows.

        .. code-block:: Python

            module = circuit.to_torch_module(semiring="log")

        As the circuit is now a regular PyTorch module, we can move it to a different device.

        .. code-block:: Python

            module = module.to("cuda:0")

    .. group-tab:: Jax

        We can turn the circuit into a jax function as follows.

        .. code-block:: Python

            func = circuit.to_jax_function(semiring="log")


The choice of semiring determines what operations the sum/product nodes perform.
By default, this is the :code:`log` semiring, which interprets sum nodes as the `logsumexp <https://en.wikipedia.org/wiki/LogSumExp>`_ operation and product nodes as addition. In the :code:`real` semiring,
product and sum nodes just compute the normal product and sum operations.

KLay doesn't introduce a batch dimension by default. So use vmap to perform batched inference.

.. tabs::

    .. code-tab:: Python PyTorch

        module = torch.vmap(module)

    .. code-tab:: Python Jax

        func = jax.vmap(func)

To achieve best runtime performance, it is advisable to use JIT compilation.

.. tabs::

    .. code-tab:: Python PyTorch

        module = torch.compile(module, mode="reduce-overhead")

    .. code-tab:: Python Jax

        func = jax.jit(func)


Klay also supports `probabilistic circuits <https://starai.cs.ucla.edu/papers/ProbCirc20.pdf>`_, which have weights associated with the edges of sum nodes.

.. tabs::

    .. code-tab:: Python PyTorch

        module2 = circuit.to_torch_module(semiring="real", probabilistic=True)

    .. code-tab:: Python Jax

        # Warning: not yet implemented!
        func2 = circuit.to_jax_module(semiring="real", probabilistic=True)


Inference
*********

The input to the circuit should be a tensor with as size the number of input literals.
Note that when using the :code:`log` semiring, the inputs are log-probabilities, while in the :code:`real` or :code:`mpe` semiring the inputs should be probabilities.
In case you are using a probabilistic circuit, you should likely have some input distributions producing these (log-)probabilities prior to the circuit.

.. tabs::

    .. code-tab:: Python PyTorch

        inputs = torch.tensor([...])
        outputs = module(inputs)

    .. code-tab:: Python Jax

        inputs = jnp.array([...])
        outputs = func(inputs)

Gradients are computed in the usual fashion.

.. tabs::

    .. code-tab:: Python PyTorch

        outputs = func(inputs)
        outputs.backward()

    .. code-tab:: Python Jax

        grad_func = jax.jit(jax.grad(func))
        grad_func(inputs)

The :code:`inputs` tensor must contain a weight for each positive literal.
The weights of the negative literals follow from those.
For example for the :code:`reals` semiring: if :code:`x` is the weight of literal :code:`l`,
then :code:`1 - x` is the weight of the negative literal :code:`-l`.
To use other weights, you must provide a separate tensor containing a weight for each negative literal.

.. tabs::

    .. code-tab:: Python PyTorch

        inputs = torch.tensor([...])
        neg_inputs = torch.tensor([...])  # assumed 1-inputs otherwise
        outputs = module(inputs, neg_inputs)

    .. code-tab:: Python Jax

        inputs = jnp.array([...])
        neg_inputs = jnp.array([...])  # assumed 1-inputs otherwise
        outputs = func(inputs, neg_inputs)