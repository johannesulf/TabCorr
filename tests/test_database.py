import numpy as np
import pytest
import tabcorr


def test_number_density(halotab, model):
    # Check that ds and wp return similar number densities. The lensing
    # has many more bins so should be more accurate. Also note that wp
    # corrects for the assumed cosmology while ds does not.

    ngal_wp = halotab["wp"].predict(model, separate_gal_type=True)[0]
    ngal_ds = halotab["ds"].predict(model, separate_gal_type=True)[0]

    cosmology = tabcorr.database.cosmology('AbacusSummit', 0)
    cosmology_obs = tabcorr.database.configuration('efficient')['cosmo_obs']

    redshift = 0.5
    rp_stretch = (
        (cosmology_obs.comoving_distance(redshift) * cosmology_obs.H0) /
        (cosmology.comoving_distance(redshift) * cosmology.H0)).value
    pi_stretch = cosmology.efunc(redshift) / cosmology_obs.efunc(redshift)
    vol_stretch = rp_stretch**2 * pi_stretch

    for gal_type in ['centrals', 'satellites']:
        assert np.isclose(ngal_wp[gal_type] * vol_stretch, ngal_ds[gal_type],
                          atol=0, rtol=1e-4)


@pytest.mark.parametrize("suite", ["AemulusAlpha", "AbacusSummit"])
def test_cosmology(suite):
    # Check that that the cosmologie work as expected.

    cosmo = tabcorr.database.cosmology(suite)

    for string in ['sigma8', 'ns', 'alphas']:
        assert string in str(cosmo)
