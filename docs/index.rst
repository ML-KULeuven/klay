.. KLay documentation master file, created by
   sphinx-quickstart on Fri Mar 14 13:16:49 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

KLay Documentation
==================

KLay is a Python library for accelerating inference on sparse arithmetic circuits.

At its core, KLay transforms sparse directed acyclic graphs into layers that can be executed in parallel.
The design of KLay is described in our paper `KLay: Accelerating Arithmetic Circuits for Neurosymbolic AI <https://arxiv.org/pdf/2410.11415>`_, published at ICLR 2025.

.. image:: _static/scatter_reduce.png
  :width: 400

----

Contents
********

.. toctree::

   Home <self>
   quickstart
   circuit_creation
   circuit_eval
   api

