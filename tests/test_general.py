import numpy as np
import pytest


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
