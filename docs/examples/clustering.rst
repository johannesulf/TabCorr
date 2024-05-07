Projected Clustering
====================

``TabCorr`` can be used to quickly predict galaxy clustering given an input halo catalog. The following example shows how to predict the projected correlation function :math:`w_{\rm p}`.

.. code-block:: python

    import numpy as np
    from halotools.mock_observables import wp
    from halotools.sim_manager import CachedHaloCatalog
    from tabcorr import TabCorr

    rp_bins = np.logspace(-1, 1, 20)
    halocat = CachedHaloCatalog(simname='bolplanck')
    halotab = TabCorr.tabulate(halocat, wp, rp_bins, pi_max=40, verbose=True,
                               num_threads=4)

Note that ``TabCorr`` by default takes into account redshift-space distortions. Now that we have calculated halo correlation functions, we can go ahead and predict galaxy correlation functions. We can even determine contributions from central-central, central-satellite, and satellite-satellite terms.

.. code-block:: python

    import matplotlib.pyplot as plt
    from halotools.empirical_models import PrebuiltHodModelFactory

    model = PrebuiltHodModelFactory('zheng07', threshold=-18)

    rp_ave = 0.5 * (rp_bins[1:] + rp_bins[:-1])

    ngal, wp = halotab.predict(model)
    plt.plot(rp_ave, wp, label='total')

    ngal, wp = halotab.predict(model, separate_gal_type=True)
    for key in wp.keys():
        plt.plot(rp_ave, wp[key], label=key, ls='--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$r_{\rm p} \ [h^{-1} \ \mathrm{Mpc}]$')
    plt.ylabel(r'$w_{\rm p} \ [h^{-1} \ \mathrm{Mpc}]$')
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout(pad=0.3)

.. image:: wp_decomposition.png
   :width: 70 %
   :align: center

Finally, we can check how the correlation functions depends on galaxy occupation parameters.

.. code-block:: python

    import matplotlib as mpl

    sm = mpl.cm.ScalarMappable(
        cmap=mpl.cm.viridis, norm=mpl.colors.Normalize(vmin=12.0, vmax=12.8))

    for logm1 in np.linspace(12.0, 12.8, 1000):
        model.param_dict['logM1'] = logm1
        ngal, wp = halotab.predict(model)
        plt.plot(rp_ave, wp, color=sm.to_rgba(logm1), lw=0.1)

    cb = plt.colorbar(sm, ax=plt.gca())
    cb.set_label(r'$\log M_1$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$r_{\rm p} \ [h^{-1} \ \mathrm{Mpc}]$')
    plt.ylabel(r'$w_{\rm p} \ [h^{-1} \ \mathrm{Mpc}]$')

.. image:: wp_vs_logm1.png
   :width: 70 %
   :align: center