import pytest
import tabcorr

from halotools.empirical_models import PrebuiltHodModelFactory


@pytest.fixture
def halotab():
    halotab = dict()
    for tpcf in ["wp", "ds", "xi0", "xi2", "xi4"]:
        halotab[tpcf] = tabcorr.database.read(
            'AbacusSummit', 0.5, tpcf, tab_config='efficient')
    return halotab


@pytest.fixture
def model():
    model = PrebuiltHodModelFactory('zheng07', prim_haloprop_key='halo_m258m',
                                    redshift=0.5)
    model.param_dict['log_eta'] = 0.0
    model.param_dict['alpha_s'] = 1.0
    model.param_dict['alpha_c'] = 0.0
    return model
