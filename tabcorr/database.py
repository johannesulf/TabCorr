"""Module providing database capabilities."""

import os
import numpy as np
from astropy import units as u
from astropy.table import Table
from . import TabCorr, Interpolator
from astropy.cosmology import Flatw0waCDM, FlatwCDM, Planck15


def configuration(config_str):
    """Describe the tabulation configuration used.

    Parameters
    ----------
    config_str : str
        String describing the configuration. You can specify a mixture of
        different configuations by separating them with a `_`. In this case,
        the first configurations in the list always take precedence in
        case multiple configurations apply to a parameter.

    Returns
    -------
    config_dict : dict
        Dictionary descibing the configuration, i.e. radial binning, number
        of satellites per halo etc.

    Raises
    ------
    ValueError
        If an unkown configuration was requested.

    """
    config_list = config_str.split('_')

    for config in config_list:
        if config not in ['aemulus', 'default']:
            raise ValueError('Unkown configuration {}.'.format(config))

    config_list.append('default')

    config_dict = {}
    config_dict['s_bins'] = {'default': np.logspace(-1.0, 1.8, 15),
                             'aemulus': np.logspace(-1, 1.78, 10)}
    config_dict['rp_wp_bins'] = {'default': np.logspace(-1.0, 1.8, 15),
                                 'aemulus': np.logspace(-1, 1.78, 10)}
    config_dict['pi_max'] = {'default': 80}
    config_dict['rp_ds_bins'] = {'default': np.logspace(-1.0, 1.8, 15)}
    config_dict['mu_bins'] = {'default': np.linspace(0, 1, 21),
                              'aemulus': np.linspace(0, 1, 41)}
    config_dict['cosmo_obs'] = {'default': Planck15, 'aemulus': None}
    config_dict['alpha_c_bins'] = {'default': np.linspace(0.0, 0.4, 4)}
    config_dict['alpha_s_bins'] = {'default': np.linspace(0.8, 1.2, 4)}
    config_dict['conc_gal_bias_bins'] = {
        'default': np.geomspace(1.0 / 3.0, 3.0, 4)}
    config_dict['sats_per_prim_haloprop'] = {'default': 2e-13}

    for parameter in config_dict.keys():
        for config in config_list:
            if config in config_dict[parameter].keys():
                config_dict[parameter] = config_dict[parameter][config]
                break

    return config_dict


def directory():
    """Return the TabCorr database directory.

    Returns
    -------
    dir : str
        The TabCorr database directory.

    Raises
    ------
    RuntimeError
        If the TABCORR_DATABASE environment variable is not set.

    """
    try:
        return os.environ['TABCORR_DATABASE']
    except KeyError:
        raise RuntimeError(
            "You must set the TABCORR_DATABASE environment variable.")


def cosmology(suite, i_cosmo=0):
    """Return the cosmology of a given simulation.

    Parameters
    ----------
    suite : str
        The simulation suite.
    i_cosmo : int, optional
        Number corresponding to the cosmology. Default is 0.

    Returns
    -------
    cosmo : str
        The TabCorr database directory.

    Raises
    ------
    ValueError
        If the an unkown simulation or cosmology number is requested.

    """
    if suite == 'AbacusSummit':
        table = Table.read(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'as_cosmos.csv'))
        table['i_cosmo'] = np.array([r[-3:] for r in table['root']], dtype=int)
        if i_cosmo not in table['i_cosmo']:
            raise ValueError('Cosmology number {} not in AbacusSummit.'.format(
                i_cosmo))
        cosmo_dict = dict(table[table['i_cosmo'] == i_cosmo][0])
        h = cosmo_dict['h']
        omega_m = cosmo_dict['omega_b'] + cosmo_dict['omega_cdm']
        n_eff = cosmo_dict['N_ur'] + cosmo_dict['N_ncdm']
        m_nu = [float(omega) * 93.04 * u.eV for omega in
                cosmo_dict['omega_ncdm'].split(',')]
        assert len(m_nu) == cosmo_dict['N_ncdm']
        while len(m_nu) < n_eff - 1:
            m_nu.append(0 * u.eV)
        return Flatw0waCDM(
            H0=h * 100, Om0=omega_m / h**2, Ob0=cosmo_dict['omega_b'] / h**2,
            w0=cosmo_dict['w0_fld'], wa=cosmo_dict['wa_fld'], Neff=n_eff,
            m_nu=m_nu, Tcmb0=2.7255 * u.K)

    elif suite == 'AemulusAlpha':
        path = os.path.dirname(os.path.realpath(__file__))
        if i_cosmo >= 0 and i_cosmo < 40:
            cosmo_dict = dict(Table.read(
                os.path.join(path, 'aa_cosmos.txt'), format='ascii')[i_cosmo])
        elif i_cosmo >= 0 and i_cosmo < 47:
            cosmo_dict = dict(Table.read(
                os.path.join(path, 'aa_test_cosmos.txt'), format='ascii')[
                    i_cosmo - 40])
        else:
            raise ValueError('Unknown cosmology number {}. '.format(i_cosmo) +
                             'Must be in the range from 0 to 46.')
        cosmo_dict['Ob0'] = cosmo_dict['ombh2'] / (cosmo_dict['H0'] / 100)**2
        cosmo_dict['Oc0'] = cosmo_dict['omch2'] / (cosmo_dict['H0'] / 100)**2
        cosmo_dict['Om0'] = cosmo_dict['Ob0'] + cosmo_dict['Oc0']
        return FlatwCDM(H0=cosmo_dict['H0'], Om0=cosmo_dict['Om0'],
                        w0=cosmo_dict['w0'], Neff=cosmo_dict['Neff'],
                        Ob0=cosmo_dict['Ob0'], Tcmb0=2.7255 * u.K)

    else:
        raise ValueError('Unkown simulation suite {}.'.format(suite))


def simulation_name(suite, i_cosmo=0, i_phase=0, config=None):
    """Return the name of a given simulation.

    Parameters
    ----------
    suite : str
        The simulation suite.
    i_cosmo : int, optional
        If applicable, number corresponding to the cosmology. Default is 0.
    i_phase : int, optional
        If applicable, number corresponding to the simulation phase. Default
        is 0.
    config : str
        Simulation configuration. Only applicable to AbacusSummit. If None,
        will default to 'base' for AbacusSummit. Default is None.

    Returns
    -------
    name : str
        The TabCorr database directory.

    Raises
    ------
    ValueError
        If the an unkown simulation, simulation number or phase number is
        requested.

    """
    if suite == 'AbacusSummit':

        if config is None:
            config = 'base'

        return '{}_c{:03d}_ph{:03d}'.format(config, i_cosmo, i_phase)

    elif suite == 'AemulusAlpha':

        if i_cosmo >= 0 and i_cosmo < 40:
            return 'Box{:03d}'.format(i_cosmo)
        elif i_cosmo >= 0 and i_cosmo < 47:
            if i_phase > 6:
                raise ValueError(
                    'Unknown phase number {}.'.format(i_phase))
            return 'TestBox{:03d}-{:03d}'.format(i_cosmo - 40, i_phase)
        else:
            raise ValueError('Unknown cosmology number {}. '.format(i_cosmo) +
                             'Must be in the range from 0 to 46.')

    else:
        raise ValueError('Unkown simulation suite {}.'.format(suite))


def simulation_snapshot_directory(
        suite, redshift, i_cosmo=0, i_phase=0, config=None):
    """Return the directory where all data for a simulation snapshot is stored.

    Parameters
    ----------
    suite : str
        The simulation suite.
    redshift : float
        The redshift of the simulation output.
    i_cosmo : int, optional
        If applicable, number corresponding to the cosmology. Default is 0.
    i_phase : int, optional
        If applicable, number corresponding to the simulation phase. Default
        is 0.
    config : str
        Simulation configuration. Only applicable to AbacusSummit. If None,
        will default to 'base' for AbacusSummit. Default is None.

    Returns
    -------
    name : str
        The directory where all data for a simulation snapshot is stored.

    """
    name = simulation_name(suite, i_cosmo=i_cosmo, i_phase=i_phase,
                           config=config)
    return os.path.join(
        directory(), suite, name,
        '{:.2f}'.format(redshift).replace('.', 'p'))


def tabcorr(suite, redshift, tpcf, i_cosmo=0, i_phase=0, sim_config=None,
            tab_config='default'):
    """Return the TabCorr tabulation for a given simulation, redshift etc.

    Parameters
    ----------
    suite : str
        The simulation suite.
    redshift : float
        The redshift of the simulation output.
    tpcf : str
        String describing the two-point correlation function.
    i_cosmo : int, optional
        If applicable, number corresponding to the cosmology. Default is 0.
    i_phase : int, optional
        If applicable, number corresponding to the simulation phase. Default
        is 0.
    sim_config : str
        Simulation configuration. Only applicable to AbacusSummit. If None,
        will default to 'base' for AbacusSummit. Default is None.
    tab_config : config_str : str
        String describing the configuration of the tabulation, i.e. binning,
        cosmology etc.

    Returns
    -------
    halotab : tabcorr.TabCorr or tabcorr.Interpolator
        The tabcorr tabulation.

    """
    directory = os.path.join(simulation_snapshot_directory(
        suite, redshift, i_cosmo=i_cosmo, i_phase=i_phase, config=sim_config),
        tab_config)

    param_dict_table = Table.read(os.path.join(
        directory, tpcf[:2] + '_grid.csv'))
    param_dict_table['log_eta'] = np.log10(param_dict_table['conc_gal_bias'])
    param_dict_table.remove_column('conc_gal_bias')
    for key in ['alpha_c', 'alpha_s', 'log_eta']:
        if len(np.unique(param_dict_table[key])) == 1:
            param_dict_table.remove_column(key)
    tabcorr_list = [TabCorr.read(os.path.join(directory, '{}_{}.hdf5'.format(
        tpcf, i))) for i in range(len(param_dict_table))]
    return Interpolator(tabcorr_list, param_dict_table)
