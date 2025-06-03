import numpy as np
import pytest
import tabcorr

from halotools.mock_observables import mean_delta_sigma as compute_ds
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import wp as compute_wp


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


def test_predictions(halotab, halocat, model):

    model.param_dict['logMmin'] = 12.9
    model.param_dict['logM1'] = 14.1
    model.param_dict['alpha'] = 1.2
    model.param_dict['log_eta'] = 0
    model.param_dict['alpha_s'] = 1.0

    model.populate_mock(halocat, Num_ptcl_requirement=299)
    ptcls = halocat.ptcl_table
    pos_p = np.column_stack([ptcls['x'], ptcls['y'], ptcls['z']])[::10]
    downsampling_factor = halocat.n_ptcls / float(len(pos_p))
    ptcl_mass = halocat.particle_mass * downsampling_factor
    config = tabcorr.database.configuration('efficient')

    rp_stretch = (
        (config['cosmo_obs'].comoving_distance(halocat.redshift) *
         config['cosmo_obs'].H0) /
        (halocat.cosmology.comoving_distance(halocat.redshift) *
         halocat.cosmology.H0))
    pi_stretch = (halocat.cosmology.efunc(halocat.redshift) /
                  config['cosmo_obs'].efunc(halocat.redshift))
    lbox_stretch = np.array([rp_stretch, rp_stretch, pi_stretch])

    wp = []
    ds = []

    for i in range(3):
        model.mock.populate()
        gals = model.mock.galaxy_table
        for xyz in ['xyz', 'yzx', 'zxy']:
            pos = return_xyz_formatted_array(
                gals[xyz[0]], gals[xyz[1]], gals[xyz[2]], period=halocat.Lbox,
                cosmology=halocat.cosmology, redshift=halocat.redshift,
                velocity=gals[f'v{xyz[2]}'], velocity_distortion_dimension='z')
            wp.append(compute_wp(
                pos * lbox_stretch, config['rp_wp_bins'], config['pi_max'],
                period=halocat.Lbox * lbox_stretch, num_threads=4))
            if xyz == 'xyz':
                ds.append(compute_ds(
                    pos, pos_p, ptcl_mass, config['rp_ds_bins'],
                    period=halocat.Lbox, num_threads=4))

    assert np.allclose(np.mean(wp, axis=0), halotab['wp'].predict(model)[1],
                       atol=0, rtol=1e-2)
    assert np.allclose(np.mean(ds, axis=0), halotab['ds'].predict(model)[1],
                       atol=0, rtol=1e-2)
