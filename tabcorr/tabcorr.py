import itertools
import h5py
import numpy as np
from scipy.spatial import Delaunay
from astropy.table import Table, vstack
from halotools.empirical_models import HodModelFactory, model_defaults
from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
from halotools.empirical_models import NFWPhaseSpace, Zheng07Sats
from halotools.mock_observables import return_xyz_formatted_array
from halotools.sim_manager import sim_defaults
from halotools.utils import crossmatch
from halotools.utils.table_utils import compute_conditional_percentiles


def print_progress(progress):
    percent = "{0:.1f}".format(100 * progress)
    bar = '=' * int(50 * progress) + '>' + ' ' * int(50 * (1 - progress))
    print('\rProgress: |{0}| {1}%'.format(bar, percent), end='\r')
    if progress == 1.0:
        print()


class TabCorr:

    def __init__(self):
        self.init = False

    @classmethod
    def tabulate(cls, halocat, tpcf, *tpcf_args,
                 mode='auto',
                 Num_ptcl_requirement=sim_defaults.Num_ptcl_requirement,
                 cosmology=sim_defaults.default_cosmology,
                 prim_haloprop_key=model_defaults.prim_haloprop_key,
                 prim_haloprop_bins=100,
                 sec_haloprop_key=model_defaults.sec_haloprop_key,
                 sec_haloprop_percentile_bins=None,
                 sats_per_prim_haloprop=3e-12, downsample=1.0,
                 verbose=False, redshift_space_distortions=True,
                 cens_prof_model=None, sats_prof_model=None, project_xyz=False,
                 cosmology_ref=None, **tpcf_kwargs):
        """
        Tabulates correlation functions for halos such that galaxy correlation
        functions can be calculated rapidly.

        Parameters
        ----------
        halocat : object
            Either an instance of `halotools.sim_manager.CachedHaloCatalog` or
            `halotools.sim_manager.UserSuppliedHaloCatalog`. This halo catalog
            is used to tabubulate correlation functions.

        tpcf : function
            The halotools correlation function for which values are tabulated.
            Positional arguments should be passed after this function.
            Additional keyword arguments for the correlation function are also
            passed through this function.

        *tpcf_args : tuple, optional
            Positional arguments passed to the ``tpcf`` function.

        mode : string, optional
            String describing whether an auto- ('auto') or a cross-correlation
            ('cross') function is going to be tabulated.

        Num_ptcl_requirement : int, optional
            Requirement on the number of dark matter particles in the halo
            catalog. The column defined by the ``prim_haloprop_key`` string
            will have a cut placed on it: all halos with
            halocat.halo_table[prim_haloprop_key] <
            Num_ptcl_requirement*halocat.particle_mass will be thrown out
            immediately after reading the original halo catalog in memory.
            Default value is set in
            `~halotools.sim_defaults.Num_ptcl_requirement`.

        cosmology : object, optional
            Instance of an astropy `~astropy.cosmology`. Default cosmology is
            set in `~halotools.sim_manager.sim_defaults`. This might be used to
            calculate phase-space distributions and redshift space distortions.

        prim_haloprop_key : string, optional
            String giving the column name of the primary halo property
            governing the occupation statistics of gal_type galaxies. Default
            value is specified in the model_defaults module.

        prim_haloprop_bins : int or list, optional
            Integer determining how many (logarithmic) bins in primary halo
            property will be used. If a list or numpy array is provided, these
            will be used as bins directly.

        sec_haloprop_key : string, optional
            String giving the column name of the secondary halo property
            governing the assembly bias. Must be a key in the table passed to
            the methods of `HeavisideAssembiasComponent`. Default value is
            specified in the `~halotools.empirical_models.model_defaults`
            module.

        sec_haloprop_percentile_bins : int, float, list or None, optional
            If an integer, it determines how many evenly spaced bins in the
            secondary halo property percentiles are going to be used. If a
            float between 0 and 1, it determines the split. Finally, if a list
            or numpy array, it directly describes the bins that are going to be
            used. If None is provided, no binning is applied.

        sats_per_prim_haloprop : float, optional
            Float determing how many satellites sample each halo. For each
            halo, the number is drawn from a Poisson distribution with an
            expectation value of ``sats_per_prim_haloprop`` times the primary
            halo property.

        downsample : float, optional
            Fraction between 0 and 1 used to downsample the total sample used
            to tabulate correlation functions. Values below unity can be used
            to reduce the computation time. It should not result in biases but
            the resulting correlation functions will be less accurate.

        verbose : boolean, optional
            Boolean determing whether the progress should be displayed.

        redshift_space_distortions : boolean, optional
            Boolean determining whether redshift space distortions should be
            applied to halos/galaxies.

        cens_prof_model : object, optional
            Instance of `halotools.empirical_models.MonteCarloGalProf` that
            determines the phase space coordinates of centrals. If none is
            provided, `halotools.empirical_models.TrivialPhaseSpace` will be
            used.

        sats_prof_model : object, optional
            Instance of `halotools.empirical_models.MonteCarloGalProf` that
            determines the phase space coordinates of satellites. If none is
            provided, `halotools.empirical_models.NFWPhaseSpace` will be used.

        project_xyz : bool, optional
            If True, the coordinates will be projected along all three spatial
            axes. By default, only the projection onto the z-axis is used.

        **tpcf_kwargs : dict, optional
                Keyword arguments passed to the ``tpcf`` function.

        Returns
        -------
        halotab : TabCorr
            Object containing all necessary information to calculate
            correlation functions for arbitrary galaxy models.
        """

        if sec_haloprop_percentile_bins is None:
            sec_haloprop_percentile_bins = np.array([0, 1])
        elif isinstance(sec_haloprop_percentile_bins, float):
            sec_haloprop_percentile_bins = np.array(
                [0, sec_haloprop_percentile_bins, 1])

        if 'period' in tpcf_kwargs:
            print('Warning: TabCorr will pass the keyword argument "period" ' +
                  'to {} based on the Lbox argument of'.format(tpcf.__name__) +
                  ' the halo catalog. The value you provided will be ignored.')
            del tpcf_kwargs['period']

        halotab = cls()

        if cosmology_ref is not None and mode == 'auto':
            rp_stretch = (
                (cosmology_ref.comoving_distance(halocat.redshift) *
                 cosmology_ref.H0) /
                (cosmology.comoving_distance(halocat.redshift) *
                 cosmology.H0))
            pi_stretch = (cosmology.efunc(halocat.redshift) /
                          cosmology_ref.efunc(halocat.redshift))
            lbox_stretch = np.array([rp_stretch, rp_stretch, pi_stretch])
        else:
            lbox_stretch = np.ones(3)

        # First, we tabulate the halo number densities.
        halos = halocat.halo_table
        halos = halos[halos['halo_pid'] == -1]
        halos = halos[halos[prim_haloprop_key] >= Num_ptcl_requirement *
                      halocat.particle_mass]
        halos[sec_haloprop_key + '_percentile'] = (
            compute_conditional_percentiles(
                table=halos, prim_haloprop_key=prim_haloprop_key,
                sec_haloprop_key=sec_haloprop_key))

        halotab.gal_type = Table()

        n_h, log_prim_haloprop_bins, sec_haloprop_percentile_bins = (
            np.histogram2d(
                np.log10(halos[prim_haloprop_key]),
                halos[sec_haloprop_key + '_percentile'],
                bins=[prim_haloprop_bins, sec_haloprop_percentile_bins]))
        halotab.gal_type['n_h'] = n_h.ravel(order='F') / np.prod(
            halocat.Lbox * lbox_stretch)

        grid = np.meshgrid(log_prim_haloprop_bins,
                           sec_haloprop_percentile_bins)
        halotab.gal_type['log_prim_haloprop_min'] = grid[0][:-1, :-1].ravel()
        halotab.gal_type['log_prim_haloprop_max'] = grid[0][:-1, 1:].ravel()
        halotab.gal_type['sec_haloprop_percentile_min'] = (
            grid[1][:-1, :-1].ravel())
        halotab.gal_type['sec_haloprop_percentile_max'] = (
            grid[1][1:, :-1].ravel())

        halotab.gal_type = vstack([halotab.gal_type, halotab.gal_type])
        halotab.gal_type['gal_type'] = np.concatenate((
            np.repeat('centrals'.encode('utf8'),
                      len(halotab.gal_type) // 2),
            np.repeat('satellites'.encode('utf8'),
                      len(halotab.gal_type) // 2)))
        halotab.gal_type['prim_haloprop'] = 10**(0.5 * (
            halotab.gal_type['log_prim_haloprop_min'] +
            halotab.gal_type['log_prim_haloprop_max']))
        halotab.gal_type['sec_haloprop_percentile'] = (0.5 * (
            halotab.gal_type['sec_haloprop_percentile_min'] +
            halotab.gal_type['sec_haloprop_percentile_max']))

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
        gals = gals[np.random.random(len(gals)) < downsample]

        idx_gals, idx_halos = crossmatch(gals['halo_id'], halos['halo_id'])
        assert np.all(gals['halo_id'][idx_gals] == halos['halo_id'][idx_halos])
        gals[sec_haloprop_key + '_percentile'] = np.zeros(len(gals))
        gals[sec_haloprop_key + '_percentile'][idx_gals] = (
            halos[sec_haloprop_key + '_percentile'][idx_halos])

        if verbose:
            print("Number of tracer particles: {0}".format(len(gals)))

        for xyz in ['xyz', 'yzx', 'zxy']:
            pos_all = return_xyz_formatted_array(
                x=gals[xyz[0]], y=gals[xyz[1]], z=gals[xyz[2]],
                velocity=gals['v'+xyz[2]] if redshift_space_distortions else 0,
                velocity_distortion_dimension='z', period=halocat.Lbox,
                redshift=halocat.redshift, cosmology=cosmology) * lbox_stretch

            pos = []
            n_gals = []
            for i in range(len(halotab.gal_type)):

                mask = (
                    (10**(halotab.gal_type['log_prim_haloprop_min'][i]) <
                     gals[prim_haloprop_key]) &
                    (10**(halotab.gal_type['log_prim_haloprop_max'][i]) >=
                     gals[prim_haloprop_key]) &
                    (halotab.gal_type['sec_haloprop_percentile_min'][i] <
                     gals[sec_haloprop_key + '_percentile']) &
                    (halotab.gal_type['sec_haloprop_percentile_max'][i] >=
                     gals[sec_haloprop_key + '_percentile']) &
                    (halotab.gal_type['gal_type'][i] == gals['gal_type']))

                pos.append(pos_all[mask])
                n_gals.append(np.sum(mask))

            n_gals = np.array(n_gals)
            n_done = 0

            if verbose:
                print("Projecting onto {0}-axis...".format(xyz[2]))

            for i in range(len(halotab.gal_type)):

                if mode == 'auto':
                    for k in range(i, len(halotab.gal_type)):
                        if len(pos[i]) * len(pos[k]) > 0:

                            if verbose:
                                n_done += (n_gals[i] * n_gals[k] * (
                                    2 if k != i else 1))
                                print_progress(n_done / np.sum(n_gals)**2)

                            xi = tpcf(
                                pos[i], *tpcf_args,
                                sample2=pos[k] if k != i else None,
                                do_auto=(i == k), do_cross=(not i == k),
                                period=halocat.Lbox * lbox_stretch,
                                **tpcf_kwargs)
                            if 'tpcf_matrix' not in locals():
                                tpcf_matrix = np.zeros(
                                    (len(xi.ravel()), len(halotab.gal_type),
                                     len(halotab.gal_type)))
                                tpcf_shape = xi.shape
                            tpcf_matrix[:, i, k] += xi.ravel()
                            tpcf_matrix[:, k, i] = tpcf_matrix[:, i, k]

                elif mode == 'cross':
                    if len(pos[i]) > 0:

                        if verbose:
                            n_done += n_gals[i]
                            print_progress(n_done / np.sum(n_gals))

                        xi = tpcf(
                            pos[i], *tpcf_args, **tpcf_kwargs,
                            period=halocat.Lbox * lbox_stretch)
                        if tpcf.__name__ == 'delta_sigma':
                            xi = xi[1]
                        if 'tpcf_matrix' not in locals():
                            tpcf_matrix = np.zeros(
                                (len(xi.ravel()), len(halotab.gal_type)))
                            tpcf_shape = xi.shape
                        tpcf_matrix[:, i] = xi.ravel()

            if not project_xyz or mode == 'cross':
                break

        if project_xyz and mode == 'auto':
            tpcf_matrix /= 3.0

        if mode == 'auto':
            tpcf_matrix_flat = []
            for i in range(tpcf_matrix.shape[0]):
                tpcf_matrix_flat.append(symmetric_matrix_to_array(
                    tpcf_matrix[i]))
            tpcf_matrix = np.array(tpcf_matrix_flat)

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
        """
        Reads tabulated correlation functions from the disk.

        Parameters
        ----------
        fname : string
            Name of the file containing the tabulated correlation functions.

        Returns
        -------
        halotab : TabCorr
            Object containing all necessary information to calculate
            correlation functions for arbitrary galaxy models.
        """

        halotab = cls()

        fstream = h5py.File(fname, 'r')
        halotab.attrs = {}
        for key in fstream.attrs.keys():
            halotab.attrs[key] = fstream.attrs[key]

        halotab.tpcf_matrix = fstream['tpcf_matrix'][()]

        halotab.tpcf_args = []
        for key in fstream['tpcf_args'].keys():
            halotab.tpcf_args.append(fstream['tpcf_args'][key][()])
        halotab.tpcf_args = tuple(halotab.tpcf_args)
        halotab.tpcf_kwargs = {}
        if 'tpcf_kwargs' in fstream:
            for key in fstream['tpcf_kwargs'].keys():
                halotab.tpcf_kwargs[key] = fstream['tpcf_kwargs'][key][()]
        halotab.tpcf_shape = tuple(fstream['tpcf_shape'][()])
        fstream.close()

        halotab.gal_type = Table.read(fname, path='gal_type')

        halotab.init = True

        return halotab

    def write(self, fname, overwrite=False, max_args_size=1000000,
              matrix_dtype=np.float32):
        """
        Writes tabulated correlation functions to the disk.

        Parameters
        ----------
        fname : string
            Name of the file that is written.

        overwrite : bool, optional
            If True, any existing file will be overwritten.

        maxsize : int, optional
            TabCorr write arguments passed to the correlation function to file.
            However, arguments that are numpy array with more entries than
            max_args_size will be omitted.
        """

        fstream = h5py.File(fname, 'w' if overwrite else 'w-')

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
        fstream.close()

        self.gal_type.write(fname, path='gal_type', append=True)

    def predict(self, model, separate_gal_type=False, **occ_kwargs):
        """
        Predicts the number density and correlation function for a certain
        model.

        Parameters
        ----------
        model : HodModelFactory
            Instance of ``halotools.empirical_models.HodModelFactory``
            describing the model for which predictions are made.

        separate_gal_type : boolean, optional
            If True, the return values are dictionaries divided by each galaxy
            types contribution to the output result.

        **occ_kwargs : dict, optional
                Keyword arguments passed to the ``mean_occupation`` functions
                of the model.

        Returns
        -------
        ngal : numpy.array or dict
            Array or dictionary of arrays containing the number densities for
            each galaxy type stored in self.gal_type. The total galaxy number
            density is the sum of all elements of this array.

        xi : numpy.array or dict
            Array or dictionary of arrays storing the prediction for the
            correlation function.
        """

        try:
            assert (sorted(model.gal_types) == sorted(
                ['centrals', 'satellites']))
        except AssertionError:
            raise RuntimeError('The model instance must only have centrals ' +
                               'and satellites as galaxy types. Check the ' +
                               'gal_types attribute of the model instance.')

        try:
            assert (model._input_model_dictionary['centrals_occupation']
                    .prim_haloprop_key == self.attrs['prim_haloprop_key'])
            assert (model._input_model_dictionary['satellites_occupation']
                    .prim_haloprop_key == self.attrs['prim_haloprop_key'])
        except AssertionError:
            raise RuntimeError('Mismatch in the primary halo properties of ' +
                               'the model and the TabCorr instance.')

        try:
            if hasattr(model._input_model_dictionary['centrals_occupation'],
                       'sec_haloprop_key'):
                assert (model._input_model_dictionary['centrals_occupation']
                        .sec_haloprop_key == self.attrs['sec_haloprop_key'])
            if hasattr(model._input_model_dictionary['satellites_occupation'],
                       'sec_haloprop_key'):
                assert (model._input_model_dictionary['satellites_occupation']
                        .sec_haloprop_key == self.attrs['sec_haloprop_key'])
        except AssertionError:
            raise RuntimeError('Mismatch in the secondary halo properties ' +
                               'of the model and the TabCorr instance.')

        try:
            assert np.abs(model.redshift - self.attrs['redshift']) < 0.05
        except AssertionError:
            raise RuntimeError('Mismatch in the redshift of the model and ' +
                               'the TabCorr instance.')

        mean_occupation = np.zeros(len(self.gal_type))

        mask = self.gal_type['gal_type'] == 'centrals'
        mean_occupation[mask] = model.mean_occupation_centrals(
            prim_haloprop=self.gal_type['prim_haloprop'][mask],
            sec_haloprop_percentile=(
                self.gal_type['sec_haloprop_percentile'][mask]), **occ_kwargs)
        mean_occupation[~mask] = model.mean_occupation_satellites(
            prim_haloprop=self.gal_type['prim_haloprop'][~mask],
            sec_haloprop_percentile=(
                self.gal_type['sec_haloprop_percentile'][~mask]), **occ_kwargs)

        ngal = mean_occupation * self.gal_type['n_h'].data

        if self.attrs['mode'] == 'auto':
            ngal_sq = np.outer(ngal, ngal)
            ngal_sq = 2 * ngal_sq - np.diag(np.diag(ngal_sq))
            ngal_sq = symmetric_matrix_to_array(ngal_sq)
            xi = self.tpcf_matrix * ngal_sq / np.sum(ngal_sq)
        elif self.attrs['mode'] == 'cross':
            xi = self.tpcf_matrix * ngal / np.sum(ngal)

        if not separate_gal_type:
            ngal = np.sum(ngal)
            xi = np.sum(xi, axis=1).reshape(self.tpcf_shape)
            return ngal, xi
        else:
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


class TabCorrInterpolation:

    def __init__(self, tabcorr_list, param_dict_table):
        """
        Initialize an interpolation of multiple TabCorr instances.

        Parameters
        ----------
        tabcorr_list : array_like
            TabCorr instances used to interpolate.

        param_dict_table : astropy.table.Table
            Table containing the keywords and values corresponding to the
            TabCorr list. Must have the same length and ordering as
            tabcorr_list.
        """

        if len(tabcorr_list) != len(param_dict_table):
            raise RuntimeError('The length of tabcorr_list does not match ' +
                               'the number of entries in param_dict_table.')

        self.keys = param_dict_table.colnames
        self.n_dim = len(self.keys)
        self.n_pts = len(param_dict_table)
        self.tabcorr_list = tabcorr_list

        if self.n_pts < self.n_dim + 1:
            raise RuntimeError('The number of TabCorr instances provided ' +
                               'must be at least the number of dimensions ' +
                               'plus 1.')

        if self.n_dim == 0:
            raise RuntimeError('param_dict_table is empty.')
        if self.n_dim == 1:
            self.x = param_dict_table.columns[0].data
        else:
            self.x = np.empty((self.n_pts, self.n_dim))
            for i, key in enumerate(param_dict_table.colnames):
                self.x[:, i] = param_dict_table[key].data
            self.delaunay = Delaunay(self.x)

    def predict(self, model, extrapolate=True, **occ_kwargs):
        """
        Linearly interpolate the predictions from multiple TabCorr instances.
        For example, this function can be used for predict correlation
        functions for continues choices of concentrations for satellites.
        Parameters to linearly interpolate over should be in the parameter
        dictionary of the model.

        Parameters
        ----------
        model : HodModelFactory
            Instance of ``halotools.empirical_models.HodModelFactory``
            describing the model for which predictions are made.

        separate_gal_type : boolean, optional
            If True, the return values are dictionaries divided by each galaxy
            types contribution to the output result.

        extrapolate : boolean, optional
            Whether to allow extrapolation beyond points sampled by x. If set
            to False, attempting to extrapolate will result in a RuntimeError.

        **occ_kwargs : dict, optional
                Keyword arguments passed to the ``mean_occupation`` functions
                of the model.

        Returns
        -------
        ngal : numpy.array or dict
            Array containing the number densities for each galaxy type stored
            in self.gal_type. The total galaxy number density is the sum of all
            elements of this array.

        xi : numpy.array or dict
            Array storing the prediction for the correlation function.
        """

        x_model = np.empty(self.n_dim)
        for i in range(self.n_dim):
            try:
                x_model[i] = model.param_dict[self.keys[i]]
            except KeyError:
                raise RuntimeError('key {} not present '.format(self.keys[i]) +
                                   'in the parameter dictionary of the model.')

        if self.n_dim > 1:

            i_simplex = self.delaunay.find_simplex(x_model)

            if i_simplex == -1:
                if not extrapolate:
                    raise RuntimeError('The parameters of the model are ' +
                                       'outside of the interpolation range ' +
                                       'and extrapolation is turned off.')
                else:
                    x_cm = np.mean(self.x[self.delaunay.simplices], axis=1)
                    i_simplex = np.argmin(np.sum((x_model - x_cm)**2, axis=1))

            simplex = self.delaunay.simplices[i_simplex]
            b = self.delaunay.transform[i_simplex, :-1].dot(
                x_model - self.delaunay.transform[i_simplex, -1])
            w = np.append(b, 1 - np.sum(b))

        else:

            if np.any(x_model < self.x) and np.any(x_model > self.x):
                simplex = [np.ma.MaskedArray.argmax(
                    np.ma.masked_array(self.x, mask=(self.x > x_model))),
                           np.ma.MaskedArray.argmin(
                    np.ma.masked_array(self.x, mask=(self.x < x_model)))]
            else:
                if not extrapolate:
                    raise RuntimeError('The parameters of the model are ' +
                                       'outside of the interpolation range ' +
                                       'and extrapolation is turned off.')
                else:
                    simplex = np.argsort(np.abs(x_model - self.x))[:2]

            w1 = (self.x[simplex[1]] - x_model) / (
                self.x[simplex[1]] - self.x[simplex[0]])
            w = [w1, 1 - w1]

        for i, k in enumerate(simplex):
            ngal_i, xi_i = self.tabcorr_list[k].predict(model, **occ_kwargs)
            if i == 0:
                ngal = ngal_i * w[i]
                xi = xi_i * w[i]
            else:
                ngal += ngal_i * w[i]
                xi += xi_i * w[i]

        return ngal, xi


def symmetric_matrix_to_array(matrix):

    try:
        assert matrix.shape[0] == matrix.shape[1]
        assert np.all(matrix == matrix.T)
    except AssertionError:
        raise RuntimeError('The matrix you provided is not symmetric.')

    n_dim = matrix.shape[0]
    sel = np.zeros((n_dim**2 + n_dim) // 2, dtype=np.int)

    for i in range(matrix.shape[0]):
        sel[(i*(i+1))//2:(i*(i+1))//2+(i+1)] = np.arange(
            i*n_dim, i*n_dim + i + 1)

    return matrix.ravel()[sel]

