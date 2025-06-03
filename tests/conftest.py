import numpy as np
import pytest
import tabcorr

from halotools.empirical_models import PrebuiltHodModelFactory
from scripts.tabulate_snapshot import read_simulation_snapshot

SUITE = 'AbacusSummit'
REDSHIFT = 0.5
COSMO = 0


@pytest.fixture
def halotab():
    halotab = dict()
    for tpcf in ["wp", "ds", "xi0", "xi2", "xi4"]:
        halotab[tpcf] = tabcorr.database.read(
            SUITE, REDSHIFT, tpcf, tab_config='efficient', i_cosmo=COSMO)
    return halotab


@pytest.fixture
def halocat():
    return read_simulation_snapshot(SUITE, REDSHIFT, COSMO)


@pytest.fixture
def model():
    model = PrebuiltHodModelFactory(
        'zheng07', redshift=0.5, prim_haloprop_key='halo_m258m',
        mdef='258m', cosmology=tabcorr.database.cosmology(SUITE, COSMO),
        concentration_bins=np.linspace(2.163, 20, 100))
    model.param_dict['log_eta'] = 0.0
    model.param_dict['alpha_s'] = 1.0
    model.param_dict['alpha_c'] = 0.0
    return model
