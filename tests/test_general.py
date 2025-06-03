import numpy as np
import pytest
import tabcorr

from scipy.interpolate import interp1d


@pytest.mark.parametrize("tpcf", ["wp", "ds"])
@pytest.mark.parametrize("interpolator", [True, False])
def test_separate_gal_type(halotab, model, tpcf, interpolator):
    # Test the consistency of separate_gal_type, i.e., the total clustering
    # is the sum of its components.

    if interpolator:
        halotab = halotab[tpcf]
    else:
        halotab = halotab[tpcf].tabcorr_list[0]

    ngal, xi = halotab.predict(model)
    ngal_sep, xi_sep = halotab.predict(model, separate_gal_type=True)

    assert len(ngal_sep) == 2
    assert len(ngal_sep) == 2 if tpcf == "ds" else 3

    assert np.isclose(ngal, np.sum([n for n in ngal_sep.values()]),
                      atol=0, rtol=1e-6)
    assert np.allclose(xi, np.sum([x for x in xi_sep.values()], axis=0),
                       atol=0, rtol=1e-6)


def test_n_gauss_prim(halotab, model):
    # Check that changing n_gauss_prim is stable.

    ngal_1, xi_1 = halotab["wp"].predict(model, n_gauss_prim=1)
    ngal_2, xi_2 = halotab["wp"].predict(model, n_gauss_prim=10)
    ngal_3, xi_3 = halotab["wp"].predict(model, n_gauss_prim=100)

    # 1 vs. 10 degrees in the Gaussian quadrature should make a difference but
    # not 10 vs. 100.
    assert not np.isclose(ngal_1, ngal_2, atol=0, rtol=1e-6)
    assert not np.allclose(xi_1, xi_2, atol=0, rtol=1e-6)
    assert np.isclose(ngal_2, ngal_3, atol=0, rtol=1e-6)
    assert np.allclose(xi_2, xi_3, atol=0, rtol=1e-6)


@pytest.mark.parametrize("tpcf", ["wp", "ds"])
def test_interpolator(halotab, model, tpcf):
    # Check that TabCorr's multi-dimensional spline interpolation agrees
    # with the one-dimensional spline interpolation of scipy.

    config = tabcorr.database.configuration('efficient')
    config['log_eta_bins'] = np.log10(config['conc_gal_bias_bins'])

    for key in ['log_eta', 'alpha_s', 'alpha_c']:
        model.param_dict['log_eta'] = 0.1
        model.param_dict['alpha_s'] = 1.1
        model.param_dict['alpha_c'] = 0.1
        bins = config[f'{key}_bins']
        xi_bins = []
        for x in bins:
            model.param_dict[key] = x
            xi_bins.append(halotab[tpcf].predict(model)[1])
        xi_bins = np.array(xi_bins)
        for x in np.linspace(np.amin(bins), np.amax(bins), 10):
            model.param_dict[key] = x
            xi_tabcorr = halotab[tpcf].predict(model)[1]
            xi_scipy = [interp1d(bins, xi_bins[:, i], kind='cubic')(x) for i in
                        range(len(xi_tabcorr))]
            assert np.allclose(xi_tabcorr, xi_scipy)
