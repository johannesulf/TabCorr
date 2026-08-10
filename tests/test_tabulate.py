import numpy as np
from halotools.custom_exceptions import HalotoolsError
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.mock_observables import return_xyz_formatted_array
from halotools.sim_manager import CachedHaloCatalog, DownloadManager

from tabcorr import TabCorr
from tabcorr.corrfunc import wp


def test_tabulate():
    # Do a full end-to-end test of the tabulation. When we use only centrals
    # and place them in every halo, the results from TabCorr and halotools
    # should agree perfectly. Here, we're not using AbacusSummit since that
    # would be too slow.

    dman = DownloadManager()
    try:
        dman.download_processed_halo_table('consuelo', 'rockstar', 0.0)
    except HalotoolsError:
        pass  # ignore if already exists

    rp_bins = np.logspace(-1, 1, 20)
    pi_max = 20
    halocat = CachedHaloCatalog(simname='consuelo')
    halotab = TabCorr.tabulate(
        halocat, wp, rp_bins, pi_max=pi_max, sats_per_prim_haloprop=1e-13,
        prim_haloprop_bins=10, n_jobs=2)

    model = PrebuiltHodModelFactory('zheng07', threshold=-18)
    model.param_dict['logMmin'] = 0
    model.param_dict['logM1'] = np.inf

    model.populate_mock(halocat)
    gals = model.mock.galaxy_table
    pos = return_xyz_formatted_array(
        gals['x'], gals['y'], gals['z'], period=halocat.Lbox,
        velocity=gals['vz'], velocity_distortion_dimension='z',
        cosmology=halocat.cosmology, redshift=halocat.redshift)

    wp_ht = wp(pos, rp_bins, pi_max, period=halocat.Lbox)
    wp_tc = halotab.predict(model)[1]

    assert np.allclose(wp_ht, wp_tc)
