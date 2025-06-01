pyDGSA -- A Python Implementation of Distance-based Generalized Sensitivity Analysis
====================================================================================

``pyDGSA`` is an open source python package designed to implement distance-based generalized
sensitivity analysis. This library provides a set of tools for calculating main effects,
interaction effects and for plotting results. For a complete description of DGSA and the theory
underpinning these analyses, please see `Fenwick et al. (2014) <https://dx.doi.org/10.1007/s11004-014-9530-5>`_ and `Park et al. (2016) <https://dx.doi.org/10.1016/j.cageo.2016.08.021>`_.

If you use the ``pyDGSA`` package in published work, please cite the paper for which it was originally developed:

Perzan, Z., Babey, T., Caers, J., Bargar, J.R. and Maher, K., 2021, Local and global sensitivity analysis of a reactive transport model simulating floodplain redox cycling, *Water Resources Research*, doi: `10.1029/2021WR029723 <https://dx.doi.org/10.1029/2021WR029723>`_

This library consists of three main submodules:

- ``dgsa`` contains the functions required to calculate the main effects and interaction effects of each parameter on model response
- ``plot`` contains several functions for plotting results, such as bar charts of sensitivity indices, bubble plots of the main effects of parameters and their interactions, and conditional cumulative distribution functions of an individual parameter
- ``cluster`` contains a Python implementation of k-medoids clustering, which is often useful for clustering model responses prior to performing DGSA

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Home page <self>
   Tutorials <tutorials>

- **Tutorials**. Most of the functionality of this package is documented in the `Tutorials <tutorials.html>`_ section. There is both a short and a detailed tutorial.
- **Documentation**. Some documentation of individual classes and functions is also included in the `API Reference <autoapi/index.html>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
