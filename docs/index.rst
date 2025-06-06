TabCorr: Tabulated Correlation Functions
########################################

.. image:: https://img.shields.io/pypi/v/tabcorr?color=blue
.. image:: https://img.shields.io/github/license/johannesulf/TabCorr?color=blue
.. image:: https://img.shields.io/github/languages/top/johannesulf/TabCorr

This Python module provides extremely efficient and precise calculations of galaxy correlation functions in halotools using tabulated values. It is specifically intended for Markov chain Monte Carlo (MCMC) exploration of the galaxy-halo connection. It implements the method described in `Zheng & Guo (2016) <https://doi.org/10.1093/mnras/stw523>`_ of tabulating correlation functions that only need to be convolved with the mean halo occupation to obtain the full correlation function of galaxies.

.. toctree::
    :maxdepth: 1
    :caption: User Guide
    :glob:

    guides/overview
    guides/database


.. toctree::
    :maxdepth: 1
    :caption: Examples
    :glob:

    examples/*

.. toctree::
    :maxdepth: 1
    :caption: API Documentation

    api

Installation
------------

The package can be installed via pip.

.. code-block:: bash

    pip install tabcorr

Author
------

Johannes Ulf Lange

Citations
---------

The method implemented in ``TabCorr`` has first been described in earlier work, particularly `Neistein et al. (2011) <https://doi.org/10.1111/j.1365-2966.2011.19145.x>`_ and `Zheng & Guo (2016) <https://doi.org/10.1093/mnras/stw523>`_. In `Lange et al. (2019a) <https://doi.org/10.1093/mnras/stz2124>`_, we developed a generalized framework for this method that also takes into account assembly bias. Finally, a good reference for the ``TabCorr`` code itself is `Lange et al. (2019b) <https://doi.org/10.1093/mnras/stz2664>`_.

License
-------

``TabCorr`` is licensed under the MIT License.

