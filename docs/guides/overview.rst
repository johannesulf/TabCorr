Overview
========

``TabCorr`` is a highly specialized Python package to predict galaxy clustering and lensing two-point correlation functions in the context of large cosmological surveys. It builds directly on the excellent ``halotools`` package. Please consult the detailed ``halotools`` `documentation <https://halotools.readthedocs.io>`_ for more background information.

``halotools`` allows us to easily make predictions for galaxy clustering and lensing properties from state-of-the-art simulations. The following code will download the Bolshoi-Planck simulation and make predictions for the projected correlation function :math:`w_\mathrm{p}` and the lensing amplitude :math:`\Delta\Sigma` using a parametric galaxy model.

.. code-block:: python

    import numpy as np

    from halotools.empirical_models import PrebuiltHodModelFactory
    from halotools.sim_manager import CachedHaloCatalog, DownloadManager
    from halotools.mock_observables import return_xyz_formatted_array, wp, mean_delta_sigma
    from time import time

    # %%

    try:
        DownloadManager().download_processed_halo_table('bolplanck', 'rockstar', 0)
    except:
        # The halo catalog was likely already downloaded.
        pass

    try:
        DownloadManager().download_ptcl_table('bolplanck', 0)
    except:
        # The particle catalog was likely already downloaded.
        pass

    halocat = CachedHaloCatalog(
        simname='bolplanck', halo_finder='rockstar', redshift=0)

    t_start = time()

    # Build the mock.
    model = PrebuiltHodModelFactory('zheng07', threshold=-19)
    model.populate_mock(halocat)

    # Define the arguments for the correlation functions.
    gals = model.mock.galaxy_table
    ptcls = model.mock.ptcl_table

    pos_g = return_xyz_formatted_array(
        gals['x'], gals['y'], gals['z'], period=halocat.Lbox,
        cosmology=halocat.cosmology, velocity=gals['vz'],
        velocity_distortion_dimension='z')
    pos_p = return_xyz_formatted_array(ptcls['x'], ptcls['y'], ptcls['z'])

    rp_bins = np.logspace(-1.0, 1.6, 14)
    pi_max = 40
    wp_args = (pos_g, rp_bins, pi_max)
    wp_kwargs = dict(period=halocat.Lbox, num_threads=4)

    downsampling_factor = halocat.num_ptcl_per_dim**3 / len(ptcls)
    m_ptcls = halocat.particle_mass * downsampling_factor
    ds_args = (pos_g, pos_p, m_ptcls, rp_bins)
    ds_kwargs = wp_kwargs

    # Calculate the correlation functions. This will take a bit.
    wp_ht = wp(*wp_args, **wp_kwargs)
    ds_ht = mean_delta_sigma(*ds_args, **ds_kwargs)

    t_end = time()
    print(f"Time taken: {t_end - t_start:.2f}s")

This calculation, despite running on a fairly small simulation and using four CPU cores, takes around a minute to run. This makes exploring the parameter space of the parametric galaxy model challenging. If we wanted to see how galaxy clustering and lensing changes under different parameters, we would have to run the entire calculation again.

This is where ``TabCorr`` comes in. The basic idea of ``TabCorr`` is that we first "tabulate" halo correlation functions. Afterward, we can quickly predict galaxy correlation functions by "convolving" the halo correlation functions with the halo occupation distribution (HOD) of the galaxy model. We refer the reader to `Zheng & Guo (2016) <https://arxiv.org/abs/1506.07523>`_ for a detailed discussion of this method.

.. code-block:: python

    import tabcorr

    halotab_wp = tabcorr.TabCorr.tabulate(
        halocat, wp, *wp_args[1:], verbose=True, **wp_kwargs)
    halotab_ds = tabcorr.TabCorr.tabulate(
        halocat, mean_delta_sigma, *ds_args[1:], mode='cross', verbose=True,
        **ds_kwargs)

While the above code block will also take a while to run, predicting galaxy clustering and lensing is now nearly instantaneous.

.. code-block:: python

    t_start = time()
    n_gal, wp_tc = halotab_wp.predict(model)
    n_gal, ds_tc = halotab_ds.predict(model)
    t_end = time()

    print(f"Time taken: {1000 * (t_end - t_start):.2f}ms")

This ``TabCorr`` prediction only takes around one millisecond. This makes it easy to explore the parameter space of the galaxy model.