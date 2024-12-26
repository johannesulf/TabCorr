"""Module implementing the halo tabulation method."""

import h5py
import tqdm
import itertools
import numpy as np
from random import shuffle
from multiprocessing import Pool
from scipy.interpolate import interp1d
from astropy.table import Table, vstack
from halotools.sim_manager import sim_defaults
from halotools.empirical_models import HodModelFactory, model_defaults
from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
from halotools.empirical_models import NFWPhaseSpace, Zheng07Sats
from halotools.mock_observables import return_xyz_formatted_array
from halotools.utils import crossmatch
from halotools.utils.table_utils import compute_conditional_percentiles


class TabCorr:
    """Class to tabulate halo and predict galaxy correlation functions."""

    @classmethod
    def tabulate(cls, halocat, tpcf, *tpcf_args,
                 mode='auto',
                 Num_ptcl_requirement=sim_defaults.Num_ptcl_requirement,
                 prim_haloprop_key=model_defaults.prim_haloprop_key,
                 prim_haloprop_bins=30,
                 sec_haloprop_key=model_defaults.sec_haloprop_key,
                 sec_haloprop_percentile_bins=None,
                 sats_per_prim_haloprop=3e-12, downsample=1.0,
                 verbose=False, redshift_space_distortions=True,
                 cens_prof_model=None, sats_prof_model=None, project_xyz=False,
                 cosmology_obs=None, num_threads=1, **tpcf_kwargs):
        """Tabulate correlation functions for halos.

        Parameters
        ----------
        halocat : halotools.sim_manager.CachedHaloCatalog or halotools.sim_manager.UserSuppliedHaloCatalog
            Halo catalog used to tabulate correlation functions.
        tpcf : function
            The halotools correlation function for which values are tabulated.
            Can also be a custom function as long as it follows the halotools
            syntax.
        *tpcf_args : tuple, optional
            Positional arguments passed to the `tpcf` function.
        mode : string, optional
            Whether an auto- ('auto') or a cross-correlation ('cross') function
            is tabulated.
        Num_ptcl_requirement : int, optional
            Requirement on the number of dark matter particles in the halo
            catalog. The column defined by the `prim_haloprop_key` string
            will have a cut placed on it: all halos with
            halocat.halo_table[prim_haloprop_key] <
            Num_ptcl_requirement*halocat.particle_mass will be thrown out.
            Default value is set in
            `halotools.sim_defaults.Num_ptcl_requirement`.
        prim_haloprop_key : string, optional
            Name of the primary halo property governing the occupation
            statistics of galaxies. Default value is specified in the
            `halotools.empirical_models.model_defaults`.
        prim_haloprop_bins : int or list, optional
            Integer determining how many (logarithmic) bins in primary halo
            property will be used. If a list or numpy array is provided, these
            will be used as bins directly. Default is 30.
        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Default value is specified in the
            `halotools.empirical_models.model_defaults` module.
        sec_haloprop_percentile_bins : int, float or None, optional
            If an integer, determines how many evenly spaced bins in the
            secondary halo property percentiles are going to be used. If a
            float between 0 and 1, it determines the split. If None is
            provided, no binning is applied. Default is None.
        sats_per_prim_haloprop : float, optional
            Determines how many satellites sample each halo. For each halo, the
            number is drawn from a Poisson distribution with an expectation
            value of `sats_per_prim_haloprop` times the primary halo property.
            Default is 3e12.
        downsample : float or function, optional
            Fraction between 0 and 1 used to downsample the total sample used
            to tabulate correlation functions. Values below unity can be used
            to reduce the computation time. It should not result in biases but
            the resulting correlation functions will be less accurate. If
            float, the same value is applied to all halos. If function, it
            should return the fraction is a function of the primary halo
            property. Default is 1.0.
        verbose : bool, optional
            Whether the progress should be displayed. Default is False.
        redshift_space_distortions : bool, optional
            Whether redshift space distortions should be applied to
            halos/galaxies. Default is True.
        cens_prof_model : object, optional
            Instance of `halotools.empirical_models.MonteCarloGalProf` that
            determines the phase space coordinates of centrals. If None,
            `halotools.empirical_models.TrivialPhaseSpace` will be used.
            Default is None.
        sats_prof_model : object, optional
            Instance of `halotools.empirical_models.MonteCarloGalProf` that
            determines the phase space coordinates of satellites. If None,
            `halotools.empirical_models.NFWPhaseSpace` will be used. Default is
            None.
        project_xyz : bool, optional
            If True, the coordinates will be projected along all three spatial
            axes. If False, only the projection onto the z-axis is used.
            Default is False.
        cosmology_obs : object, optional
            Instance of an astropy `astropy.cosmology`. This can be used to
            correct coordinates in the simulation for the Alcock-Paczynski (AP)
            effect, i.e. a mismatch between the cosmology of the model
            (simulation) and the cosmology used to interpret observations. Note
            that the cosmology of the simulation is part of the halocat object.
            If None, no correction for the AP effect is applied. Also, a
            correction for the AP effect is only applied for auto-correlation
            functions. Default is None.
        num_threads : int, optional
            How many threads to use for the tabulation. Default is 1.
        **tpcf_kwargs : dict, optional
            Keyword arguments passed to the `tpcf` function.

        Returns
        -------
        halotab : tabcorr.TabCorr
            `TabCorr` object.

        Raises
        ------
        ValueError
            If invalid halo bins are given.
        RuntimeError
            If TabCorr encounters an internal error.

        """
        if 'period' in tpcf_kwargs:
            print('Warning: TabCorr will pass the keyword argument "period" ' +
                  'to {} based on the Lbox argument of'.format(tpcf.__name__) +
                  ' the halo catalog. The value you provided will be ignored.')
            del tpcf_kwargs['period']

        halotab = cls()

        if cosmology_obs is not None and mode == 'auto':
            rp_stretch = (
                (cosmology_obs.comoving_distance(halocat.redshift) *
                 cosmology_obs.H0) /
                (halocat.cosmology.comoving_distance(halocat.redshift) *
                 halocat.cosmology.H0))
            pi_stretch = (halocat.cosmology.efunc(halocat.redshift) /
                          cosmology_obs.efunc(halocat.redshift))
            lbox_stretch = np.array([rp_stretch, rp_stretch, pi_stretch])
        else:
            lbox_stretch = np.ones(3)

        # First, we tabulate the halo number densities.
        halos = halocat.halo_table
        halos = halos[halos['halo_upid'] == -1]
        halos = halos[halos[prim_haloprop_key] >
                      Num_ptcl_requirement * halocat.particle_mass]

        if isinstance(prim_haloprop_bins, int):
            log_prim_haloprop_bins = np.linspace(
                np.log10(np.amin(halos[prim_haloprop_key])) - 1e-3,
                np.log10(np.amax(halos[prim_haloprop_key])) + 1e-3,
                prim_haloprop_bins + 1)
        elif isinstance(log_prim_haloprop_bins, (list, np.ndarray)):
            log_prim_haloprop_bins = prim_haloprop_bins
        else:
            raise ValueError('prim_haloprop_bins must be an int, list or ' +
                             'numpy array.')

        if sec_haloprop_percentile_bins is None:
            sec_haloprop_percentile_bins = np.array([-1e-3, 1 + 1e-3])
        elif isinstance(sec_haloprop_percentile_bins, float):
            if not (0 < sec_haloprop_percentile_bins and
                    sec_haloprop_percentile_bins < 1):
                raise ValueError('sec_haloprop_percentile_bins must be ' +
                                 'between 0 and 1.')
            sec_haloprop_percentile_bins = np.array(
                [-1e-3, sec_haloprop_percentile_bins, 1 + 1e-3])
        elif isinstance(sec_haloprop_percentile_bins, int):
            sec_haloprop_percentile_bins = np.linspace(
                -1e-3, 1 + 1e-3, sec_haloprop_percentile_bins + 1)
        else:
            raise ValueError('sec_haloprop_percentile_bins must be an int, ' +
                             'float, list or numpy array.')

        halos[sec_haloprop_key + '_percentile'] = (
            compute_conditional_percentiles(
                table=halos, prim_haloprop_key=prim_haloprop_key,
                sec_haloprop_key=sec_haloprop_key))

        halotab.gal_type = Table()

        n_h, log_prim_haloprop_bins, sec_haloprop_percentile_bins = (
            np.histogram2d(
                np.log10(halos[prim_haloprop_key]),
                halos[sec_haloprop_key + '_percentile'],
                bins=[log_prim_haloprop_bins, sec_haloprop_percentile_bins]))
        halotab.gal_type['n_h'] = n_h.ravel(order='F')

        grid = np.meshgrid(log_prim_haloprop_bins,
                           sec_haloprop_percentile_bins)
        halotab.gal_type['log_prim_haloprop_min'] = grid[0][:-1, :-1].ravel()
        halotab.gal_type['log_prim_haloprop_max'] = grid[0][:-1, 1:].ravel()
        halotab.gal_type['sec_haloprop_percentile_min'] = (
            grid[1][:-1, :-1].ravel())
        halotab.gal_type['sec_haloprop_percentile_max'] = (
            grid[1][1:, :-1].ravel())
        halotab.gal_type['prim_haloprop'] = 10**(0.5 * (
            halotab.gal_type['log_prim_haloprop_min'] +
            halotab.gal_type['log_prim_haloprop_max']))
        halotab.gal_type['sec_haloprop_percentile'] = (0.5 * (
            halotab.gal_type['sec_haloprop_percentile_min'] +
            halotab.gal_type['sec_haloprop_percentile_max']))
        prim_haloprop = sort_into_bins(
            np.log10(halos[prim_haloprop_key]), log_prim_haloprop_bins,
            halos[sec_haloprop_key + '_percentile'],
            sec_haloprop_percentile_bins, halos[prim_haloprop_key])
        halotab.gal_type['prim_haloprop_dist_index'] = np.zeros(
            len(halotab.gal_type))
        for i in range(len(halotab.gal_type)):
            if len(prim_haloprop[i]) > 0:
                x_min = 10**halotab.gal_type['log_prim_haloprop_min'][i]
                x_max = 10**halotab.gal_type['log_prim_haloprop_max'][i]
                x_mean = np.mean(prim_haloprop[i])
                halotab.gal_type['prim_haloprop_dist_index'][i] =\
                    distribution_index(x_min, x_max, x_mean)

        halotab.gal_type = vstack([halotab.gal_type, halotab.gal_type])
        halotab.gal_type['gal_type'] = np.concatenate((
            np.repeat('centrals'.encode('utf8'),
                      len(halotab.gal_type) // 2),
            np.repeat('satellites'.encode('utf8'),
                      len(halotab.gal_type) // 2)))

        # Now, we tabulate the correlation functions.
        cens_occ_model = Zheng07Cens(prim_haloprop_key=prim_haloprop_key)
        if cens_prof_model is None:
            cens_prof_model = TrivialPhaseSpace(redshift=halocat.redshift)
        sats_occ_model = Zheng07Sats(prim_haloprop_key=prim_haloprop_key)
        if sats_prof_model is None:
            sats_prof_model = NFWPhaseSpace(redshift=halocat.redshift)

        model = HodModelFactory(
            centrals_occupation=cens_occ_model,
            centrals_profile=cens_prof_model,
            satellites_occupation=sats_occ_model,
            satellites_profile=sats_prof_model)

        model.param_dict['logMmin'] = 0
        model.param_dict['sigma_logM'] = 0.1
        model.param_dict['alpha'] = 1.0
        model.param_dict['logM0'] = 0
        model.param_dict['logM1'] = - np.log10(sats_per_prim_haloprop)
        model.populate_mock(halocat, Num_ptcl_requirement=Num_ptcl_requirement)

        gals = model.mock.galaxy_table
        idx_gals, idx_halos = crossmatch(gals['halo_id'], halos['halo_id'])
        assert np.all(gals['halo_id'][idx_gals] == halos['halo_id'][idx_halos])
        gals[sec_haloprop_key + '_percentile'] = np.zeros(len(gals))
        gals[sec_haloprop_key + '_percentile'][idx_gals] = (
            halos[sec_haloprop_key + '_percentile'][idx_halos])

        if verbose:
            print("Number of tracer particles: {0}".format(len(gals)))

        for xyz in ['xyz', 'yzx', 'zxy']:

            if verbose and project_xyz:
                print("Projecting onto {0}-axis...".format(xyz[2]))

            pos = (return_xyz_formatted_array(
                x=gals[xyz[0]], y=gals[xyz[1]], z=gals[xyz[2]],
                velocity=gals['v'+xyz[2]] if redshift_space_distortions else 0,
                velocity_distortion_dimension='z', period=halocat.Lbox,
                redshift=halocat.redshift, cosmology=halocat.cosmology) *
                lbox_stretch)

            period = halocat.Lbox * lbox_stretch

            pos = sort_into_bins(
                np.log10(gals[prim_haloprop_key]), log_prim_haloprop_bins,
                gals[sec_haloprop_key + '_percentile'],
                sec_haloprop_percentile_bins, pos,
                gal_type=gals['gal_type'])

            assert len(pos) == len(halotab.gal_type)

            for i in range(len(halotab.gal_type)):

                if halotab.gal_type['gal_type'][i] == 'centrals':
                    # Make sure the number of halos are consistent.
                    try:
                        assert len(pos[i]) == int(halotab.gal_type['n_h'][i])
                    except AssertionError:
                        raise RuntimeError('There was an internal error in ' +
                                           'TabCorr. If possible, please ' +
                                           'report this bug in the TabCorr ' +
                                           'GitHub repository.')
                else:
                    if len(pos[i]) == 0 and halotab.gal_type['n_h'][i] != 0:
                        raise RuntimeError(
                            'There was at least one bin without satellite ' +
                            'tracers. Increase sats_per_prim_haloprop.')

                if len(pos[i]) > 0:

                    if isinstance(downsample, float):
                        select = np.random.random(len(pos[i])) < downsample
                    else:
                        select = (
                            np.random.random(len(pos[i])) <
                            downsample(halotab.gal_type['prim_haloprop'][i]))

                    # If the down-sampling reduced the number of tracers to at
                    # or below one, force at least 2 tracers to not bias the
                    # clustering estimates.
                    if np.sum(select) <= 1 and len(pos[i]) > 1:
                        select = np.zeros(len(pos[i]), dtype=bool)
                        select[np.random.choice(len(pos[i]), size=2)] = True

                    pos[i] = pos[i][select]

            if xyz == 'xyz':
                tpcf_matrix, tpcf_shape = compute_tpcf_matrix(
                    mode, pos, tpcf, period, tpcf_args, tpcf_kwargs,
                    num_threads=num_threads, verbose=verbose)

            if not project_xyz or mode == 'cross':
                break
            elif xyz != 'xyz':
                tpcf_matrix += compute_tpcf_matrix(
                    mode, pos, tpcf, period, tpcf_args, tpcf_kwargs,
                    num_threads=num_threads, verbose=verbose)[0]

        if project_xyz and mode == 'auto':
            tpcf_matrix /= 3.0

        if mode == 'auto':
            tpcf_matrix_flat = []
            for i in range(tpcf_matrix.shape[0]):
                tpcf_matrix_flat.append(symmetric_matrix_to_array(
                    tpcf_matrix[i]))
            tpcf_matrix = np.array(tpcf_matrix_flat)

        # Remove entries that don't have any halos.
        use = halotab.gal_type['n_h'] != 0
        halotab.gal_type = halotab.gal_type[use]
        if mode == 'auto':
            use = symmetric_matrix_to_array(np.outer(use, use))
        tpcf_matrix = tpcf_matrix[:, use]

        # Convert into number densities.
        halotab.gal_type['n_h'] /= np.prod(halocat.Lbox * lbox_stretch)

        halotab.attrs = {}
        halotab.attrs['tpcf'] = tpcf.__name__
        halotab.attrs['mode'] = mode
        halotab.attrs['simname'] = halocat.simname
        halotab.attrs['redshift'] = halocat.redshift
        halotab.attrs['Num_ptcl_requirement'] = Num_ptcl_requirement
        halotab.attrs['prim_haloprop_key'] = prim_haloprop_key
        halotab.attrs['sec_haloprop_key'] = sec_haloprop_key

        halotab.tpcf_args = tpcf_args
        halotab.tpcf_kwargs = tpcf_kwargs
        halotab.tpcf_shape = tpcf_shape
        halotab.tpcf_matrix = tpcf_matrix

        halotab.init = True

        return halotab

    @classmethod
    def read(cls, fname):
        """Read tabulated correlation functions from the disk.

        Parameters
        ----------
        fname : string or h5py.Group
            Name of the file or h5py group containing the TabCorr object.

        Returns
        -------
            `TabCorr` object.

        """
        halotab = cls()

        if not isinstance(fname, h5py.Group):
            fstream = h5py.File(fname, 'r')
        else:
            fstream = fname

        halotab.attrs = {}
        for key in fstream.attrs.keys():
            halotab.attrs[key] = fstream.attrs[key]

        halotab.tpcf_matrix = fstream['tpcf_matrix'][()].astype(np.float64)

        halotab.tpcf_args = []
        for key in fstream['tpcf_args'].keys():
            halotab.tpcf_args.append(fstream['tpcf_args'][key][()])
        halotab.tpcf_args = tuple(halotab.tpcf_args)
        halotab.tpcf_kwargs = {}
        if 'tpcf_kwargs' in fstream:
            for key in fstream['tpcf_kwargs'].keys():
                halotab.tpcf_kwargs[key] = fstream['tpcf_kwargs'][key][()]
        halotab.tpcf_shape = tuple(fstream['tpcf_shape'][()])

        if not isinstance(fname, h5py.Group):
            fstream.close()

        halotab.gal_type = Table.read(fname, path='gal_type')

        return halotab

    def write(self, fname, overwrite=False, max_args_size=1000000,
              matrix_dtype=np.float32):
        """Write the tabulated correlation functions to the disk.

        Parameters
        ----------
        fname : string or h5py.Group
            Name of the file or h5py group the data is written to.
        overwrite : bool, optional
            If True, any existing file will be overwritten. Default is False.
        max_args_size : int, optional
            By default, TabCorr writes all arguments passed to the correlation
            function when calling `tabulate` to file. However, arguments that
            are numpy arrays with more entries than max_args_size will be
            omitted. Default is 1000000.
        matrix_dtype : type
            The dtype used to write the correlation matrix to disk. Can be used
            to save space at the expense of precision.

        """
        if not isinstance(fname, h5py.Group):
            fstream = h5py.File(fname, 'w' if overwrite else 'w-')
        else:
            fstream = fname

        keys = ['tpcf', 'mode', 'simname', 'redshift', 'Num_ptcl_requirement',
                'prim_haloprop_key', 'sec_haloprop_key']
        for key in keys:
            fstream.attrs[key] = self.attrs[key]

        fstream['tpcf_matrix'] = self.tpcf_matrix.astype(matrix_dtype)

        for i, arg in enumerate(self.tpcf_args):
            if (type(arg) is not np.ndarray or
                    np.prod(arg.shape) < max_args_size):
                fstream['tpcf_args/arg_%d' % i] = arg
        for key in self.tpcf_kwargs:
            if (type(self.tpcf_kwargs[key]) is not np.ndarray or
                    np.prod(self.tpcf_kwargs[key].shape) < max_args_size):
                fstream['tpcf_kwargs/' + key] = self.tpcf_kwargs[key]
        fstream['tpcf_shape'] = self.tpcf_shape

        if not isinstance(fname, h5py.Group):
            fstream.close()

        self.gal_type.write(fname, path='gal_type', append=True)

    def mean_occupation(self, model, n_gauss_prim=10, check_consistency=True,
                        **occ_kwargs):
        """Calculate the mean occupation for each halo/galaxy bin.

        Parameters
        ----------
        model : HodModelFactory
            Instance of ``halotools.empirical_models.HodModelFactory``
            describing the model for which predictions are made.
        n_gauss_prim : int, optional
            The number of points used in the Gaussian quadrature to calculate
            the mean occupation averaged over the primary halo property in each
            halo bin. Default is 10.
            Whether to enforce consistency in the redshift, primary halo
            property and secondary halo property between the model and the
            TabCorr instance. Default is True.
        **occ_kwargs : dict, optional
            Keyword arguments passed to the ``mean_occupation`` functions of
            the model.

        Returns
        -------
        n : numpy.ndarray
            Array containing the mean occuaption numbers. Has the same length
            as `self.gal_type`.

        Raises
        ------
        ValueError
            If there is a mis-match between the model and the TabCorr instance.
        """
        if check_consistency:
            try:
                assert (sorted(model.gal_types) == sorted(
                    ['centrals', 'satellites']))
            except AssertionError:
                raise ValueError(
                    'The model instance must only have centrals and ' +
                    'satellites as galaxy types. Check the `gal_types` ' +
                    'attribute of the model instance.')
            try:
                assert (model._input_model_dictionary['centrals_occupation']
                        .prim_haloprop_key == self.attrs['prim_haloprop_key'])
                assert (model._input_model_dictionary['satellites_occupation']
                        .prim_haloprop_key == self.attrs['prim_haloprop_key'])
            except AssertionError:
                raise ValueError('Mismatch in the primary halo properties ' +
                                 'of the model and the TabCorr instance.')

            try:
                if hasattr(
                        model._input_model_dictionary['centrals_occupation'],
                        'sec_haloprop_key'):
                    assert (
                        model._input_model_dictionary['centrals_occupation']
                        .sec_haloprop_key == self.attrs['sec_haloprop_key'])
                if hasattr(
                        model._input_model_dictionary['satellites_occupation'],
                        'sec_haloprop_key'):
                    assert (
                        model._input_model_dictionary['satellites_occupation']
                        .sec_haloprop_key == self.attrs['sec_haloprop_key'])
            except AssertionError:
                raise ValueError('Mismatch in the secondary halo properties ' +
                                 'of the model and the TabCorr instance.')

            try:
                assert np.abs(model.redshift - self.attrs['redshift']) < 0.05
            except AssertionError:
                raise ValueError('Mismatch in the redshift of the model and ' +
                                 'the TabCorr instance.')

        log_prim_haloprop_min = self.gal_type['log_prim_haloprop_min'].data
        log_prim_haloprop_max = self.gal_type['log_prim_haloprop_max'].data
        d_log_prim_haloprop = log_prim_haloprop_max - log_prim_haloprop_min
        sec_haloprop_percentile = self.gal_type['sec_haloprop_percentile'].data
        gal_type = self.gal_type['gal_type']

        if not hasattr(self, 'x_gauss') or len(self.x_gauss) != n_gauss_prim:
            self.x_gauss, self.w_gauss = np.polynomial.legendre.leggauss(
                n_gauss_prim)
            self.x_gauss = (self.x_gauss + 1) / 2

        prim_haloprop = 10**(log_prim_haloprop_min + d_log_prim_haloprop *
                             self.x_gauss[:, np.newaxis]).T.ravel()
        sec_haloprop_percentile = np.repeat(sec_haloprop_percentile,
                                            n_gauss_prim)
        gal_type = np.repeat(gal_type, n_gauss_prim)

        mean_occupation = np.zeros(len(prim_haloprop))
        select = gal_type == 'centrals'
        mean_occupation[select] = model.mean_occupation_centrals(
            prim_haloprop=prim_haloprop[select],
            sec_haloprop_percentile=sec_haloprop_percentile[select],
            **occ_kwargs)
        mean_occupation[~select] = model.mean_occupation_satellites(
            prim_haloprop=prim_haloprop[~select],
            sec_haloprop_percentile=sec_haloprop_percentile[~select],
            **occ_kwargs)
        mean_occupation = mean_occupation.reshape(
            (len(self.gal_type), n_gauss_prim))
        prim_haloprop = prim_haloprop.reshape(mean_occupation.shape)

        if 'prim_haloprop_dist_index' in self.gal_type.colnames:
            # Add +1 to the index since we integrate over log M, not M.
            n = self.gal_type['prim_haloprop_dist_index'][:, np.newaxis] + 1
        else:
            # Ignore the halo mass distribution if the TabCorr object was
            # tabulated with an earlier version of TabCorr.
            n = 0

        return (np.sum(
            self.w_gauss * mean_occupation * prim_haloprop**n, axis=-1) /
            np.sum(self.w_gauss * prim_haloprop**n, axis=-1))

    def predict(self, model, separate_gal_type=False, n_gauss_prim=10,
                check_consistency=True, **occ_kwargs):
        """Predict the number density and correlation function for a model.

        Parameters
        ----------
        model : HodModelFactory or numpy.ndarray
            Instance of ``halotools.empirical_models.HodModelFactory``
            describing the model for which predictions are made or a numpy
            array containing the mean occupation in each halo bin. The latter
            option is mainly used internally.
        separate_gal_type : bool, optional
            If True, the return values are dictionaries divided by each galaxy
            types contribution to the output result. Default is False.
        n_gauss_prim : int, optional
            The number of points used in the Gaussian quadrature to calculate
            the mean occupation averaged over the primary halo property in each
            halo bin. Default is 10.
        check_consistency: bool, optional
            Whether to enforce consistency in the redshift, primary halo
            property and secondary halo property between the model and the
            TabCorr instance. Default is True.
        **occ_kwargs : dict, optional
            Keyword arguments passed to the ``mean_occupation`` functions of
            the model.

        Returns
        -------
        ngal : float or dict
            Galaxy number density. If `separate_gal_type` is True, this is a
            dictionary splitting contributions by galaxy type.
        xi : numpy.ndarray
            Correlation function values. If `separate_gal_type` is True, this
            is a dictionary splitting contributions by galaxy type.

        """
        if not isinstance(model, np.ndarray):
            mean_occupation = self.mean_occupation(
                model, n_gauss_prim=n_gauss_prim,
                check_consistency=check_consistency, **occ_kwargs)
        else:
            mean_occupation = model

        ngal = mean_occupation * self.gal_type['n_h'].data

        if self.attrs['mode'] == 'auto':
            if not hasattr(self, 'ngal_sq_index_1'):
                n_bins = len(self.gal_type)
                ngal_sq_index_1 = np.repeat(np.arange(n_bins), n_bins).reshape(
                    n_bins, n_bins)
                ngal_sq_index_2 = np.tile(np.arange(n_bins), n_bins).reshape(
                    n_bins, n_bins)

                self.ngal_sq_index_1 = symmetric_matrix_to_array(
                    ngal_sq_index_1, check_symmetry=False)
                self.ngal_sq_index_2 = symmetric_matrix_to_array(
                    ngal_sq_index_2, check_symmetry=False)

                self.ngal_sq_prefactor = np.where(
                    self.ngal_sq_index_1 == self.ngal_sq_index_2, 1, 2)

            ngal_sq = (self.ngal_sq_prefactor * ngal[self.ngal_sq_index_1] *
                       ngal[self.ngal_sq_index_2])

        if not separate_gal_type:
            if self.attrs['mode'] == 'auto':
                xi = (np.einsum('ij, j', self.tpcf_matrix, ngal_sq) /
                      np.sum(ngal_sq))
            elif self.attrs['mode'] == 'cross':
                xi = np.einsum('ij, j', self.tpcf_matrix, ngal) / np.sum(ngal)
            return np.sum(ngal), xi.reshape(self.tpcf_shape)

        if self.attrs['mode'] == 'auto':
            xi = (self.tpcf_matrix * ngal_sq) / np.sum(ngal_sq)
        elif self.attrs['mode'] == 'cross':
            xi = (self.tpcf_matrix * ngal) / np.sum(ngal)

        ngal_dict = {}
        xi_dict = {}

        for gal_type in np.unique(self.gal_type['gal_type']):
            mask = self.gal_type['gal_type'] == gal_type
            ngal_dict[gal_type] = np.sum(ngal[mask])

        if self.attrs['mode'] == 'auto':
            for gal_type_1, gal_type_2 in (
                    itertools.combinations_with_replacement(
                        np.unique(self.gal_type['gal_type']), 2)):
                mask = symmetric_matrix_to_array(np.outer(
                    gal_type_1 == self.gal_type['gal_type'],
                    gal_type_2 == self.gal_type['gal_type']) |
                    np.outer(
                    gal_type_2 == self.gal_type['gal_type'],
                    gal_type_1 == self.gal_type['gal_type']))
                xi_dict['%s-%s' % (gal_type_1, gal_type_2)] = np.sum(
                    xi * mask, axis=1).reshape(self.tpcf_shape)

        elif self.attrs['mode'] == 'cross':
            for gal_type in np.unique(self.gal_type['gal_type']):
                mask = self.gal_type['gal_type'] == gal_type
                xi_dict[gal_type] = np.sum(
                    xi * mask, axis=1).reshape(self.tpcf_shape)

        return ngal_dict, xi_dict


def sort_into_bins(log_prim_haloprop, log_prim_haloprop_bins,
                   sec_haloprop_percentile, sec_haloprop_percentile_bins,
                   x, gal_type=None):
    """Sort an input array `x` into bins used by `TabCorr`.

    Parameters
    ----------
    log_prim_haloprop : numpy.ndarray
        Primary halo property. Must have the same length at `x`.
    log_prim_haloprop_bins : numpy.ndarray
        Primary halo property bins.
    sec_haloprop_percentile : numpy.ndarray
        Secondary halo property percentiles. Must have the same length at `x`.
    sec_haloprop_percentile_bins : numpy.ndarray
        Secondary halo property percentile bins.
    x : numpy.ndarray
        Array to sort into bins.
    gal_type : None or numpy.ndarray, optional
        Galaxy types. If None, results are not sorted by galaxy type. Default
        is None.

    Returns
    -------
    x_sorted : list
        Values of `x` sorted into bins.

    """
    n_p = len(log_prim_haloprop_bins) - 1
    n_s = len(sec_haloprop_percentile_bins) - 1

    i_prim = np.digitize(
        log_prim_haloprop, bins=log_prim_haloprop_bins, right=False) - 1
    i_sec = np.digitize(
        sec_haloprop_percentile, bins=sec_haloprop_percentile_bins,
        right=False) - 1
    # Throw out those that don't fall into any bin.
    x = x[~((i_prim < 0) | (i_prim >= n_p) | (i_sec < 0) | (i_sec >= n_s))]

    if gal_type is not None:
        i_type = np.where(gal_type == 'centrals', 0, 1)
        n_t = 2
    else:
        i_type = 0
        n_t = 1

    i = (i_prim + i_sec * n_p + i_type * n_p * n_s)
    x_sorted = x[np.argsort(i)]
    counts = np.bincount(i, minlength=n_p * n_s * n_t)
    counts = np.cumsum(counts)
    counts = np.insert(counts, 0, 0)

    return [x_sorted[counts[i]:counts[i+1]] for i in range(len(counts) - 1)]


def distribution_index(x_min, x_max, x_mean):
    r"""Calculate an effective power-law index :math:`n`.

    :math:`n` is defined such that if :math:`p(x) = x^n` and is integrated over
    `x_min` to `x_max`, it gives a certain mean `x_mean`.

    Parameters
    ----------
    x_min : float
        Lower bound of the integration range.
    x_max : float
        Upper bound of the integration range.
    x_mean : float
        Mean of the distribution.

    Returns
    -------
    n : float
        Index :math:`n` to reproduce the mean, but not smaller than -10 or
        larger than +10.
    """
    x_max = x_max / x_min
    x_mean = x_mean / x_min
    n_interp = np.linspace(-10, +10, 100)
    x_interp = ((n_interp + 1) / (n_interp + 2) * (x_max**(n_interp + 2) - 1) /
                (x_max**(n_interp + 1) - 1))
    return interp1d(x_interp, n_interp, kind='cubic', fill_value=(-10, +10),
                    bounds_error=False)(x_mean)


def symmetric_matrix_to_array(matrix, check_symmetry=True):
    """Reduce a symmetric 2-dimensional matrix into 1-dimensional array.

    Parameters
    ----------
    matrix : numpy.ndarray
        Symmetric matrix
    check_symmetry : bool, optional
        Whether to check that `matrix` describes a symmetric matrix. Default is
        True.

    Returns
    -------
    m_array : numpy.ndarray
        Array containing all unique values of `matrix`.

    Raises
    ------
    ValueError
        If `matrix` is not a symmetric matrix and `check_symmetry` is True.

    """
    if check_symmetry:
        try:
            assert matrix.shape[0] == matrix.shape[1]
            assert np.all(matrix == matrix.T)
        except AssertionError:
            raise ValueError('The matrix you provided is not symmetric.')

    n_dim = matrix.shape[0]
    sel = np.zeros((n_dim**2 + n_dim) // 2, dtype=int)

    for i in range(matrix.shape[0]):
        sel[(i*(i+1))//2:(i*(i+1))//2+(i+1)] = np.arange(
            i*n_dim, i*n_dim + i + 1)

    return matrix.ravel()[sel]


GLOBAL_ARGS = {}


def compute_tpcf(i):
    """Compute the two-point correlation for a given sample of tracers.

    Parameters
    ----------
    i : int or tuple
        Which  samples to calculate the two-point correlation function for.

    Returns
    -------
    i : int or tuple
        Which samples the two-point correlation function was calculated for.
    xi : numpy.ndarray
        The two-point correlation function.

    """
    mode = GLOBAL_ARGS['mode']
    pos = GLOBAL_ARGS['pos']
    tpcf = GLOBAL_ARGS['tpcf']
    period = GLOBAL_ARGS['period']
    tpcf_args = GLOBAL_ARGS['tpcf_args']
    tpcf_kwargs = GLOBAL_ARGS['tpcf_kwargs']

    if mode == 'auto':
        i_1, i_2 = i
        if len(pos[i_1]) > len(pos[i_2]):
            i_1, i_2 = i_2, i_1
        return i, tpcf(pos[i_1], *tpcf_args, sample2=pos[i_2] if i_1 != i_2
                       else None,  do_auto=(i_1 == i_2), do_cross=(i_1 != i_2),
                       period=period, **tpcf_kwargs)
    else:
        return i, tpcf(pos[i], *tpcf_args, period=period, **tpcf_kwargs)


def compute_tpcf_matrix(mode, pos, tpcf, period, tpcf_args, tpcf_kwargs,
                        num_threads=1, verbose=False):
    """Calculate the two-point correlation function matrix between all samples.

    Parameters
    ----------
    mode : string
        Whether an auto- ('auto') or a cross-correlation ('cross') function is
        be tabulated.
    pos : numpy.ndarray
        Samples.
    period : numpy.ndarray
        Box size.
    tpcf : function
        The halotools correlation function for which values are tabulated. Can
        also be a custom function as long as it follows the halotools syntax.
    tpcf_args : tuple, optional
        Positional arguments passed to the `tpcf` function.
    tpcf_kwargs : dict, optional
        Keyword arguments passed to the `tpcf` function.
    num_threads : int, optional
        How many threads to use for the tabulation. Default is 1.
    verbose : bool, optional
        Whether the progress should be displayed. Default is False.

    Returns
    -------
    tpcf_matrix : numpy.ndarray
        Two-point correlation function matrix. Two-dimensional if `mode` is
        'cross' and three-dimensional if `mode` is 'auto'.
    tpcf_shape : numpy.ndarray
        Shape of the two-point correlation function returned by `tpcf`.

    """
    global GLOBAL_ARGS
    GLOBAL_ARGS['mode'] = mode
    GLOBAL_ARGS['pos'] = pos
    GLOBAL_ARGS['tpcf'] = tpcf
    GLOBAL_ARGS['period'] = period
    GLOBAL_ARGS['tpcf_args'] = tpcf_args
    GLOBAL_ARGS['tpcf_kwargs'] = tpcf_kwargs

    tasks = [i for i in range(len(pos)) if len(pos[i]) > 0]

    if mode == 'auto':
        tasks = list(itertools.combinations_with_replacement(tasks, 2))

    shuffle(tasks)

    if verbose:
        pbar = tqdm.tqdm(total=len(tasks), smoothing=0)

    tpcf_matrix = None

    with Pool(num_threads) as pool:
        for i, xi in pool.imap_unordered(compute_tpcf, tasks):

            if tpcf_matrix is None:
                if mode == 'auto':
                    tpcf_matrix = np.zeros(
                        (len(xi.ravel()), len(pos), len(pos)))
                else:
                    tpcf_matrix = np.zeros((len(xi.ravel()), len(pos)))

            if mode == 'auto':
                i_1, i_2 = i
                tpcf_matrix[:, i_1, i_2] += xi.ravel()
                tpcf_matrix[:, i_2, i_1] = tpcf_matrix[:, i_1, i_2]
            else:
                tpcf_matrix[:, i] += xi.ravel()

            if verbose:
                pbar.update(1)

            tpcf_shape = xi.shape

    return tpcf_matrix, tpcf_shape
