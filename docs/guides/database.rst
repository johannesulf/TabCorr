TabCorr Database
================

The tabulation of the halo correlation functions can be computationally demanding. Therefore, many results are available as part of the TabCorr database. You can interact with the data from within TabCorr via the :py:meth:`tabcorr.database` module. For this to work, the environment variable ``TABCORR_DATABASE`` needs to be set. This can be done via ``export TABCORR_DATABASE=<PATH>``, where ``<PATH>`` is the path containing the database folders, e.g. the `AemulusAlpha` folder. See the following code snippet for basic usage.

Example
-------

.. code-block:: python

    from tabcorr import database
    from halotools.empirical_models import PrebuiltHodModelFactory
    
    # Load the TabCorr tabulation for the redshift-space monopole for the first
    # simulation, B00, of the AemulusAlpha simulation suite at redshift 0.25.
    halotab = database.read('AemulusAlpha', 0.25, 'xi0', i_cosmo=0,
                            tab_config='default')
    
    # Get information about the tabulation, i.e., the radial bins.
    s_bins = database.configuration('default')['s_bins']

    # Get the cosmology of that simulation.
    cosmo = database.cosmology('AemulusAlpha', i_cosmo=0)
    
    # Build the model.
    model = PrebuiltHodModelFactory(
        'zheng07', prim_haloprop_key='halo_m200m', redshift=0.25)
    
    # Add the phase-space parameters in addition to the occupation parameters of
    # the Zheng07 occupation model. See Lange et al. (2023) for a definition of
    # these parameters. Note that log_eta is the logarithm base 10 of the parameter
    # conc_gal_bias in BiasedNFWPhaseSpace.
    model.param_dict['alpha_c'] = 0
    model.param_dict['alpha_s'] = 1.0
    model.param_dict['log_eta'] = 0

    # Predict the clustering.
    halotab.predict(model)

Products
--------

The following data products are avilable on `zenodo <https://zenodo.org/records/15588541>`_. This database may be expanded in the future. If you're interested in additional simulations, redshifts, or other configurations, please open an issue on the `TabCorr Github page <https://github.com/johannesulf/TabCorr/issues>`_.

.. list-table::
    :header-rows: 1

    * - Suite
      - Simulation
      - Cosmologies
      - Phases
      - Redshifts
      - Statistics
      - Configurations
    * - AemulusAlpha
      - —
      - 0-39
      - —
      - 0.25, 0.40
      - :math:`w_{\rm p}`, :math:`\xi_{0, 2, 4}`, :math:`\Delta\Sigma`
      - default
    * - AemulusAlpha
      - —
      - 0-39
      - —
      - 0.55
      - :math:`w_{\rm p}`, :math:`\xi_{0, 2, 4}`
      - aemulus
    * - AbacusSummit
      - Base
      - all\ [1]_
      - 0
      - 0.5
      - :math:`w_{\rm p}`, :math:`\xi_{0, 2, 4}`, :math:`\Delta\Sigma`
      - efficient
    * - AbacusSummit
      - Base
      - all\ [1]_ :math:`\Lambda`\ CDM with :math:`\sum m_\nu = 0.06 \, \mathrm{eV}`
      - 0
      - 0.2, 0.3, 0.4, 0.8\ [2]_
      - :math:`w_{\rm p}`, :math:`\Delta\Sigma`
      - efficient
    * - AbacusSummit
      - High
      - 0
      - 100
      - 0.2, 0.3, 0.4, 0.5, 0.8
      - :math:`w_{\rm p}`, :math:`\Delta\Sigma`
      - efficient

.. [1] Cosmologies 11, 21, and 22 do not have any data.
.. [2] Cosmology 103 is missing data for redshift 0.8.

Please have a look at the ``scripts`` folder in the `GitHub repository <https://github.com/johannesulf/TabCorr/tree/main/scripts>`_, particularly ``parse_snapshot.py`` and ``tabulate_snapshot.py``, to understand how these are generated.

Attribution
-----------

If you're using the data from AemulusAlpha in your published work, you must cite `DeRose et al. (2019) <https://doi.org/10.3847/1538-4357/ab1085>`_ for creating these simulations. Similarly, if you use AbacusSummit, please cite `Maksimova et al. (2021) <https://doi.org/10.1093/mnras/stab2484>`_, `Garrison et al. (2021) <https://doi.org/10.1093/mnras/stab2482>`_, and `Hadzhiyska et al. (2021) <https://doi.org/10.1093/mnras/stab2980>`_. In all cases, I would appreciate a reference to `Lange et al. (2023) <https://doi.org/10.1093/mnras/stad473>`_ where I created the tabulated correlation functions.
