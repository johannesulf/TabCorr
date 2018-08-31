import h5py
import numpy as np
import itertools
from astropy.table import Table, vstack
from halotools.empirical_models import PrebuiltHodModelFactory, model_defaults
from halotools.mock_observables import return_xyz_formatted_array
from halotools.sim_manager import sim_defaults
from halotools.utils import crossmatch
from halotools.utils.table_utils import compute_conditional_percentiles


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
                 **tpcf_kwargs):
        r"""
        Tabulates correlation functions for halos such that galaxy correlation
        functions can be calculated rapidly.

        Parameters
        ----------
        halocat : object
            Either an instance of `~halotools.sim_manager.CachedHaloCatalog` or
            `~halotools.sim_manager.UserSuppliedHaloCatalog`. This halo catalog
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

        *tpcf_kwargs : dict, optional
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

        halotab = cls()

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
        halotab.gal_type['n_h'] = n_h.ravel(order='F') / np.prod(halocat.Lbox)

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
        model = PrebuiltHodModelFactory('zheng07', redshift=halocat.redshift,
                                        prim_haloprop_key=prim_haloprop_key)
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

        pos_all = return_xyz_formatted_array(
            x=gals['x'], y=gals['y'], z=gals['z'],
            velocity=gals['vz'] if redshift_space_distortions else 0,
            velocity_distortion_dimension='z', period=halocat.Lbox,
            redshift=halocat.redshift, cosmology=cosmology)

        pos = []
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

        for i in range(len(halotab.gal_type)):

            if verbose:
                print("row %d/%d" % (i + 1, len(halotab.gal_type)))

            if mode == 'auto':
                for k in range(i, len(halotab.gal_type)):
                    if len(pos[i]) * len(pos[k]) > 0:
                        xi = tpcf(
                            pos[i], *tpcf_args,
                            sample2=pos[k] if k != i else None,
                            do_auto=(i == k), do_cross=(not i == k),
                            **tpcf_kwargs)
                        if 'tpcf_matrix' not in locals():
                            tpcf_matrix = np.zeros(
                                (len(xi.ravel()), len(halotab.gal_type),
                                 len(halotab.gal_type)))
                            tpcf_shape = xi.shape
                        tpcf_matrix[:, i, k] = xi.ravel()
                        tpcf_matrix[:, k, i] = xi.ravel()

            elif mode == 'cross':
                if len(pos[i]) > 0:
                    xi = tpcf(
                        pos[i], *tpcf_args, **tpcf_kwargs)
                    if tpcf.__name__ == 'delta_sigma':
                        xi = xi[1]
                    if 'tpcf_matrix' not in locals():
                        tpcf_matrix = np.zeros(
                            (len(xi.ravel()), len(halotab.gal_type)))
                        tpcf_shape = xi.shape
                    tpcf_matrix[:, i] = xi.ravel()

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
        r"""
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

        halotab.tpcf_matrix = fstream['tpcf_matrix'].value
        halotab.tpcf_args = []
        for key in fstream['tpcf_args'].keys():
            halotab.tpcf_args.append(fstream['tpcf_args'][key].value)
        halotab.tpcf_args = tuple(halotab.tpcf_args)
        halotab.tpcf_kwargs = {}
        if 'tpcf_kwargs' in fstream:
            for key in fstream['tpcf_kwargs'].keys():
                halotab.tpcf_kwargs[key] = fstream['tpcf_kwargs'][key].value
        halotab.tpcf_shape = tuple(fstream['tpcf_shape'].value)
        fstream.close()

        halotab.gal_type = Table.read(fname, path='gal_type')

        halotab.init = True

        return halotab

    def write(self, fname):
        r"""
        Writes tabulated correlation functions to the disk.

        Parameters
        ----------
        fname : string
            Name of the file that is written.
        """

        fstream = h5py.File(fname, 'w-')

        keys = ['tpcf', 'mode', 'simname', 'redshift', 'Num_ptcl_requirement',
                'prim_haloprop_key', 'sec_haloprop_key']
        for key in keys:
            fstream.attrs[key] = self.attrs[key]

        fstream['tpcf_matrix'] = self.tpcf_matrix
        for i, arg in enumerate(self.tpcf_args):
            fstream['tpcf_args/arg_%d' % i] = arg
        for key in self.tpcf_kwargs:
            fstream['tpcf_kwargs/' + key] = self.tpcf_kwargs[key]
        fstream['tpcf_shape'] = self.tpcf_shape
        fstream.close()

        self.gal_type.write(fname, path='gal_type', append=True)

    def predict(self, model, separate_gal_type=False):
        r"""
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
                self.gal_type['sec_haloprop_percentile'][mask]))
        mean_occupation[~mask] = model.mean_occupation_satellites(
            prim_haloprop=self.gal_type['prim_haloprop'][~mask],
            sec_haloprop_percentile=(
                self.gal_type['sec_haloprop_percentile'][~mask]))

        ngal = mean_occupation * self.gal_type['n_h'].data

        if self.attrs['mode'] == 'auto':
            xi = self.tpcf_matrix * np.outer(ngal, ngal) / np.sum(ngal)**2
        elif self.attrs['mode'] == 'cross':
            xi = self.tpcf_matrix * ngal / np.sum(ngal)

        if not separate_gal_type:
            ngal = np.sum(ngal)
            if self.attrs['mode'] == 'auto':
                xi = np.sum(xi, axis=(1, 2)).reshape(self.tpcf_shape)
            elif self.attrs['mode'] == 'cross':
                xi = np.sum(xi, axis=1).reshape(self.tpcf_shape)
            return ngal, xi
        else:
            ngal_dict = {}
            xi_dict = {}

            for gal_type in np.unique(self.gal_type['gal_type']):
                mask = self.gal_type['gal_type'] == gal_type
                ngal_dict[gal_type] = np.sum(ngal[mask])

            if self.attrs['mode'] == 'auto':
                grid = np.meshgrid(self.gal_type['gal_type'],
                                   self.gal_type['gal_type'])
                for gal_type_1, gal_type_2 in (
                        itertools.combinations_with_replacement(
                            np.unique(self.gal_type['gal_type']), 2)):
                    mask = (((gal_type_1 == grid[0]) &
                             (gal_type_2 == grid[1])) |
                            ((gal_type_1 == grid[1]) &
                             (gal_type_2 == grid[0])))
                    xi_dict['%s-%s' % (gal_type_1, gal_type_2)] = np.sum(
                        xi * mask, axis=(1, 2)).reshape(self.tpcf_shape)

            elif self.attrs['mode'] == 'cross':
                for gal_type in np.unique(self.gal_type['gal_type']):
                    mask = self.gal_type['gal_type'] == gal_type
                    xi_dict[gal_type] = np.sum(
                        xi * mask, axis=1).reshape(self.tpcf_shape)

            return ngal_dict, xi_dict
