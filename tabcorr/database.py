"""Module providing database capabilities."""

import os
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.cosmology import Flatw0waCDM, FlatwCDM, Parameter, Planck15
from astropy.table import Table

from .interpolator import Interpolator


def configuration(name):
    """Describe the tabulation configuration used.

    Parameters
    ----------
    name : str
        String describing the configuration. You can specify a mixture of
        different configuations by separating them with a `_`. In this case,
        the first configurations in the list always take precedence in
        case multiple configurations apply to a parameter.

    Returns
    -------
    config : dict
        Dictionary descibing the configuration, i.e. radial binning, number
        of satellites per halo etc.

    Raises
    ------
    ValueError
        If an unkown configuration was requested.

    """
    choices = name.split('_')
    choices.append('default')

    for choice in choices:
        if choice not in ['aemulus', 'default', 'efficient']:
            raise ValueError(f"Unkown configuration '{choice}'.")

    config = {}
    config['s_bins'] = {'default': np.logspace(-1.0, 1.8, 15),
                        'aemulus': np.logspace(-1, 1.78, 10)}
    config['rp_wp_bins'] = {'default': np.logspace(-1.0, 1.8, 15),
                            'aemulus': np.logspace(-1, 1.78, 10)}
    config['pi_max'] = {'default': 80}
    config['rp_ds_bins'] = {'default': np.logspace(-1.0, 1.8, 15),
                            'efficient': np.logspace(-1.0, 1.6, 14)}
    config['mu_bins'] = {'default': np.linspace(0, 1, 21),
                         'aemulus': np.linspace(0, 1, 41)}
    config['cosmo_obs'] = {'default': Planck15, 'aemulus': None}
    config['alpha_c_bins'] = {'default': np.linspace(0.0, 0.4, 4)}
    config['alpha_s_bins'] = {'default': np.linspace(0.8, 1.2, 4)}
    config['conc_gal_bias_bins'] = {
        'default': np.geomspace(1.0 / 3.0, 3.0, 4)}
    config['sats_per_prim_haloprop'] = {'default': 2e-13,
                                        'efficient': 1e-13}
    config['downsample'] = {'default': 1.0,
                            'efficient': (lambda x: x / 1e13)}

    for key, options in config.items():
        for choice in choices:
            if choice in options:
                config[key] = options[choice]
                break

    return config


class TabCorrCosmology:

    sigma8 = Parameter(fvalidate='float')
    ns = Parameter(fvalidate='float')
    alphas = Parameter(fvalidate='float')

    def __init__(self, *args, sigma8=0.8, ns=0.96, alphas=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma8 = sigma8
        self.ns = ns
        self.alphas = alphas


class TabCorrFlatwCDM(TabCorrCosmology, FlatwCDM):
    pass


class TabCorrFlatw0waCDM(TabCorrCosmology, Flatw0waCDM):
    pass


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
    cosmo : astropy.cosmology.Cosmology
        The cosmology associated with the simulation. This is a regular
        astropy cosmology object with `sigma8` added as an extra attribute.

    Raises
    ------
    ValueError
        If the an unkown simulation or cosmology number is requested.

    """
    if suite == 'AbacusSummit':
        table = Table.read(Path(__file__).absolute().parent / 'as_cosmos.csv')
        table['i_cosmo'] = np.array([r[-3:] for r in table['root']], dtype=int)
        if i_cosmo not in table['i_cosmo']:
            msg = f"Cosmology number {i_cosmo} not in AbacusSummit."
            raise ValueError(msg)
        cosmo_dict = dict(table[table['i_cosmo'] == i_cosmo][0])
        h = cosmo_dict['h']
        omega_m = cosmo_dict['omega_b'] + cosmo_dict['omega_cdm']
        n_eff = cosmo_dict['N_ur'] + cosmo_dict['N_ncdm']
        m_nu = [float(omega) * 93.04 * u.eV for omega in
                cosmo_dict['omega_ncdm'].split(',')]
        assert len(m_nu) == max(cosmo_dict['N_ncdm'], 1)
        while len(m_nu) < n_eff - 1:
            m_nu.append(0 * u.eV)
        return TabCorrFlatw0waCDM(
            H0=h * 100, Om0=omega_m / h**2, Ob0=cosmo_dict['omega_b'] / h**2,
            w0=cosmo_dict['w0_fld'], wa=cosmo_dict['wa_fld'], Neff=n_eff,
            m_nu=m_nu, Tcmb0=2.7255 * u.K, sigma8=cosmo_dict['sigma8_cb'],
            ns=cosmo_dict['n_s'], alphas=cosmo_dict['alpha_s'])

    elif suite == 'AemulusAlpha':
        path = Path(__file__).absolute().parent
        if i_cosmo >= 0 and i_cosmo < 40:
            cosmo_dict = dict(Table.read(
                path / 'aa_cosmos.txt', format='ascii')[i_cosmo])
        elif i_cosmo >= 0 and i_cosmo < 47:
            cosmo_dict = dict(Table.read(path / 'aa_test_cosmos.txt',
                                         format='ascii')[i_cosmo - 40])
        else:
            msg = ("Unknown cosmology number {i_cosmo}. Must be in the range "
                   "from 0 to 46.")
            raise ValueError(msg)
        cosmo_dict['Ob0'] = cosmo_dict['ombh2'] / (cosmo_dict['H0'] / 100)**2
        cosmo_dict['Oc0'] = cosmo_dict['omch2'] / (cosmo_dict['H0'] / 100)**2
        cosmo_dict['Om0'] = cosmo_dict['Ob0'] + cosmo_dict['Oc0']
        return TabCorrFlatwCDM(
            H0=cosmo_dict['H0'], Om0=cosmo_dict['Om0'], w0=cosmo_dict['w0'],
            Neff=cosmo_dict['Neff'], Ob0=cosmo_dict['Ob0'],
            Tcmb0=2.7255 * u.K, sigma8=cosmo_dict['sigma8'],
            ns=cosmo_dict['ns'])
    else:
        msg = f"Unkown simulation suite {suite}."
        raise ValueError(msg)


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
    config : str, optional
        Simulation configuration. Only applicable to AbacusSummit. If None,
        will default to 'base' for AbacusSummit. Default is None.

    Returns
    -------
    name : str
        The name of the simulation.

    Raises
    ------
    ValueError
        If the an unkown simulation, simulation number or phase number is
        requested.

    """
    if suite == 'AbacusSummit':

        if config is None:
            config = 'base'

        return f'{config}_c{i_cosmo:03d}_ph{i_phase:03d}'

    elif suite == 'AemulusAlpha':

        if i_cosmo >= 0 and i_cosmo < 40:
            return f'Box{i_cosmo:03d}'
        elif i_cosmo >= 0 and i_cosmo < 47:
            if i_phase > 6:
                msg = f"Unknown phase number {i_phase}."
                raise ValueError(msg)
            return f'TestBox{i_cosmo - 40:03d}-{i_phase:03d}'
        else:
            msg = (f"Unknown cosmology number {i_cosmo}. Must be in the range "
                   "from 0 to 46.")
            raise ValueError(msg)

    else:
        msg = f"Unkown simulation suite {suite}."
        raise ValueError(msg)


def directory(suite, redshift, i_cosmo=0, i_phase=0, config=None):
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
    config : str, optional
        Simulation configuration. Only applicable to AbacusSummit. If None,
        will default to 'base' for AbacusSummit. Default is None.

    Returns
    -------
    path : pathlib.Path
        The directory where all data for a simulation snapshot is stored.

    Raises
    ------
    RuntimeError
        If the TABCORR_DATABASE environment variable is not set.

    """
    try:
        path = Path(os.environ['TABCORR_DATABASE'])
    except KeyError:
        msg = "You must set the TABCORR_DATABASE environment variable."
        raise RuntimeError(msg)
    name = simulation_name(suite, i_cosmo=i_cosmo, i_phase=i_phase,
                           config=config)
    return (path / suite / name / f'{redshift:.2f}'.replace('.', 'p'))


def read(suite, redshift, tpcf, i_cosmo=0, i_phase=0, sim_config=None,
         tab_config='default'):
    """Read the TabCorr tabulation for a given simulation, redshift etc.

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
    sim_config : str, optional
        Simulation configuration. Only applicable to AbacusSummit. If None,
        will default to 'base' for AbacusSummit. Default is None.
    tab_config : config_str : str, optional
        String describing the configuration of the tabulation, i.e. binning,
        cosmology etc.

    Returns
    -------
    halotab : tabcorr.TabCorr or tabcorr.Interpolator
        The tabcorr tabulation.

    """
    path = directory(
        suite, redshift, i_cosmo=i_cosmo, i_phase=i_phase, config=sim_config)

    return Interpolator.read(path / f'{tpcf}_{tab_config}.hdf5')


# Define an alias for backwards compatibility.
tabcorr = read
