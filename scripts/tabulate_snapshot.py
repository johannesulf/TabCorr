import os
import copy
import argparse
import numpy as np
import multiprocessing
import astropy.units as u
from astropy.table import Table
from astropy.constants import G
from tabcorr import TabCorr, database
from tabcorr.corrfunc import wp, s_mu_tpcf
from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.mock_observables import tpcf_multipole, mean_delta_sigma
from halotools.empirical_models import TrivialPhaseSpace, BiasedNFWPhaseSpace


def read_simulation_snapshot(
        suite, redshift, i_cosmo=0, i_phase=None, config=None):
    name = database.simulation_name(
        suite, i_cosmo=i_cosmo, i_phase=i_phase, config=config)
    directory = database.simulation_snapshot_directory(
        suite, redshift, i_cosmo=i_cosmo, i_phase=i_phase, config=config)
    halos = Table.read(os.path.join(directory, 'snapshot.hdf5'),
                       path='halos')
    try:
        ptcls = Table.read(os.path.join(directory, 'snapshot.hdf5'),
                           path='particles')
    except OSError:
        ptcls = None
    cosmology = database.cosmology(suite, i_cosmo=i_cosmo)

    if suite == 'AbacusSummit':
        mdef = '{:.0f}m'.format(halos.meta['SODensityL1'])
        lbox = halos.meta['BoxSize']
        particle_mass = halos.meta['ParticleMassHMsun']
        halo_vmax = None
    else:
        mdef = '200m'
        lbox = 1050
        particle_mass = 3.51e10 * cosmology.Om0 / 0.3
        halo_vmax = halos['halo_vmax']

    return UserSuppliedHaloCatalog(
        redshift=redshift, Lbox=lbox, particle_mass=particle_mass,
        simname=name, halo_x=halos['halo_x'], halo_y=halos['halo_y'],
        halo_z=halos['halo_z'], halo_vx=halos['halo_vx'],
        halo_vy=halos['halo_vy'], halo_vz=halos['halo_vz'],
        halo_id=np.arange(len(halos)), halo_pid=np.repeat(-1, len(halos)),
        halo_upid=np.repeat(-1, len(halos)),
        halo_nfw_conc=halos['halo_r{}'.format(mdef)] / halos['halo_rs'],
        halo_mvir=halos['halo_m{}'.format(mdef)],
        halo_rvir=halos['halo_r{}'.format(mdef)] * 1e-9,
        halo_hostid=np.arange(len(halos)), cosmology=cosmology,
        halo_vmax=halo_vmax,
        **{'halo_m{}'.format(mdef): halos['halo_m{}'.format(mdef)],
           'halo_r{}'.format(mdef): halos['halo_r{}'.format(mdef)]}), ptcls


class ScaledBiasedNFWPhaseSpace(BiasedNFWPhaseSpace):

    def __init__(self, profile_integration_tol=1e-5, **kwargs):
        BiasedNFWPhaseSpace.__init__(
            self, profile_integration_tol=profile_integration_tol, **kwargs)
        self.param_dict['alpha_s'] = 1.0

    def _vrad_disp_from_lookup(self, scaled_radius, *concentration_array,
                               **kwargs):
        return (BiasedNFWPhaseSpace._vrad_disp_from_lookup(
                self, scaled_radius, *concentration_array, **kwargs) *
                self.param_dict['alpha_s'])


class CentralVelocitBiasPhaseSpace(TrivialPhaseSpace):

    def __init__(self, mdef='200m', **kwargs):
        TrivialPhaseSpace.__init__(self, **kwargs)
        self.param_dict['alpha_c'] = 0.0
        self.mdef = mdef

    def assign_phase_space(self, table, **kwargs):
        TrivialPhaseSpace.assign_phase_space(self, table, **kwargs)
        vscale = np.sqrt(
            G * table['halo_m' + self.mdef] * u.Msun / (
                table['halo_r' + self.mdef] * u.Mpc / (1 + self.redshift))).to(
                    u.km / u.s).value
        for key in ['vx', 'vy', 'vz']:
            table[key][:] += (vscale * np.random.normal(size=len(table)) *
                              self.param_dict['alpha_c'] / np.sqrt(3.0))


def tabcorr_s_mu_to_multipole(halotab_s_mu, mu_bins, order):
    halotab_mult = copy.deepcopy(halotab_s_mu)
    halotab_mult.tpcf_shape = (halotab_s_mu.tpcf_shape[0], )
    halotab_mult.tpcf_matrix = np.zeros(
        (halotab_s_mu.tpcf_shape[0], halotab_s_mu.tpcf_matrix.shape[1]))

    for i in range(halotab_s_mu.tpcf_matrix.shape[1]):
        halotab_mult.tpcf_matrix[:, i] = tpcf_multipole(
            halotab_s_mu.tpcf_matrix[:, i].reshape(
                halotab_s_mu.tpcf_shape), mu_bins, order=order)

    return halotab_mult


def main():

    parser = argparse.ArgumentParser(
        description='Tabulate halo correlation functions.')
    parser.add_argument('suite', help='simulation suite',
                        choices=['AemulusAlpha'])
    parser.add_argument('redshift', help='simulation redshift', type=float)
    parser.add_argument('--cosmo', help='simulation cosmology, default is 0',
                        type=int, default=0)
    parser.add_argument('--phase', help='simulation phase, default is 0',
                        type=int, default=0)
    parser.add_argument('--sim_config', default=None,
                        help='simulation configuration to assume')
    parser.add_argument('--tab_config', default='default',
                        help='tabulation configuration to assume')
    parser.add_argument('--tpcf', default='xi', choices=['xi', 'wp', 'ds'],
                        help='TPCF to tabulate')

    args = parser.parse_args()

    config = database.configuration(args.tab_config)

    halocat, ptcls = read_simulation_snapshot(
            args.suite, args.redshift, i_cosmo=args.cosmo, i_phase=args.phase,
            config=args.sim_config)

    for key in halocat.halo_table.colnames:
        if key[:6] == 'halo_m' and key[-1] == 'm':
            mdef = key[6:]

    if args.tpcf == 'wp' and config['pi_max'] >= 80:
        config['alpha_c_bins'] = [0.0]

    if args.tpcf == 'ds':
        config['alpha_c_bins'] = [0.0]
        config['alpha_s_bins'] = [1.0]

    path = os.path.join(database.simulation_snapshot_directory(
        args.suite, args.redshift, i_cosmo=args.cosmo, i_phase=args.phase,
        config=args.sim_config), args.tab_config)
    if not os.path.isdir(path):
        os.makedirs(path)

    phase_space_grid = np.array(np.meshgrid(
        config['alpha_c_bins'], config['alpha_s_bins'],
        config['conc_gal_bias_bins'])).T.reshape(-1, 3)
    table = Table()
    table['alpha_c'] = phase_space_grid[:, 0]
    table['alpha_s'] = phase_space_grid[:, 1]
    table['conc_gal_bias'] = phase_space_grid[:, 2]
    table.write(os.path.join(path, args.tpcf + '_grid.csv'), overwrite=True)

    for i, (alpha_c, alpha_s, conc_gal_bias) in enumerate(phase_space_grid):

        cens_prof_model = CentralVelocitBiasPhaseSpace(
            redshift=halocat.redshift, mdef=mdef)
        cens_prof_model.param_dict['alpha_c'] = alpha_c

        sats_prof_model = ScaledBiasedNFWPhaseSpace(
            mdef=mdef, redshift=halocat.redshift, cosmology=halocat.cosmology,
            concentration_bins=np.linspace(2.163, 20, 100),
            conc_gal_bias_bins=np.array([conc_gal_bias]))
        sats_prof_model.param_dict['alpha_s'] = alpha_s

        if args.tpcf == 'ds':
            prim_haloprop_bins = 300
            mode = 'cross'
        else:
            prim_haloprop_bins = 30
            mode = 'auto'

        prim_haloprop_key = 'halo_m' + mdef

        if args.suite == 'AbacusSummit':
            sec_haloprop_key = 'halo_nfw_conc'
        else:
            sec_haloprop_key = 'halo_vmax'

        sec_haloprop_percentile_bins = 0.5

        if args.suite == 'AbacusSummit':
            num_ptcl_requirement = 299
        else:
            num_ptcl_requirement = 99

        kwargs = {'mode': mode, 'cens_prof_model': cens_prof_model,
                  'sats_prof_model': sats_prof_model, 'verbose': False,
                  'num_threads': multiprocessing.cpu_count(),
                  'sats_per_prim_haloprop': config['sats_per_prim_haloprop'],
                  'project_xyz': True,
                  'prim_haloprop_bins': prim_haloprop_bins,
                  'prim_haloprop_key': prim_haloprop_key,
                  'sec_haloprop_key': sec_haloprop_key,
                  'sec_haloprop_percentile_bins': sec_haloprop_percentile_bins,
                  'cosmology_obs': config['cosmo_obs'],
                  'Num_ptcl_requirement': num_ptcl_requirement}

        if args.tpcf == 'xi':
            halotab_s_mu = TabCorr.tabulate(
                halocat, s_mu_tpcf, config['s_bins'], config['mu_bins'],
                **kwargs)
            for order in [0, 2, 4]:
                halotab_multipole = tabcorr_s_mu_to_multipole(
                    halotab_s_mu, config['mu_bins'], order)
                halotab_multipole.write(os.path.join(
                    path, 'xi{}_{}.hdf5'.format(order, i)), overwrite=True)

        elif args.tpcf == 'wp':
            halotab = TabCorr.tabulate(
                halocat, wp, config['rp_wp_bins'], config['pi_max'], **kwargs)
            halotab.write(os.path.join(path, 'wp_{}.hdf5'.format(i)),
                          overwrite=True)

        elif args.tpcf == 'ds':

            ptcl_pos = np.vstack([ptcls['x'], ptcls['y'], ptcls['z']]).T

            if args.suite == 'AemulusAlpha':
                n_ptcl_tot = 1400**3

            downsampling_factor = n_ptcl_tot / float(len(ptcl_pos))
            ptcl_mass = halocat.particle_mass * downsampling_factor

            halotab = TabCorr.tabulate(
                halocat, mean_delta_sigma, ptcl_pos, ptcl_mass,
                config['rp_ds_bins'], **kwargs)
            halotab.write(os.path.join(path, 'ds_{}.hdf5'.format(i)),
                          overwrite=True)


if __name__ == "__main__":
    main()
