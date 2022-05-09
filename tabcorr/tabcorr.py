import h5py
import tqdm
import time
import itertools
import numpy as np
from queue import Empty
from random import shuffle
from multiprocessing import Process, Queue
from scipy.spatial import Delaunay
from astropy.table import Table, vstack
from halotools.sim_manager import sim_defaults
from halotools.empirical_models import HodModelFactory, model_defaults
from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
from halotools.empirical_models import NFWPhaseSpace, Zheng07Sats
from halotools.mock_observables import return_xyz_formatted_array
from halotools.utils import crossmatch
from halotools.utils.table_utils import compute_conditional_percentiles


def compute_tpcf(mode, pos, tpcf, period, tpcf_args, tpcf_kwargs,
                 input_queue, output_queue):

    while True:
        try:
            i = input_queue.get(block=True, timeout=0.1)
            if mode == 'auto':
                i_1, i_2 = i
                xi = tpcf(pos[i_1], *tpcf_args, sample2=pos[i_2] if i_1 != i_2
                          else None,  do_auto=(i_1 == i_2),
                          do_cross=(i_1 != i_2), period=period, **tpcf_kwargs)
            else:
                xi = tpcf(pos[i], *tpcf_args, period=period, **tpcf_kwargs)

            output_queue.put((i, xi))

        except Empty:
            break


def compute_tpcf_matrix(mode, pos, tpcf, period, tpcf_args, tpcf_kwargs,
                        num_threads=1, verbose=False):

    tpcf_matrix = None

    input_queue = Queue()
    output_queue = Queue()

    if mode == 'auto':
        tasks = itertools.combinations_with_replacement(range(len(pos)), 2)
    else:
        tasks = range(len(pos))

    tasks = list(tasks)
    shuffle(tasks)

    n_tot = 0
    n = 0

    for task in tasks:
        if mode == 'auto':
            i_1, i_2 = task
            if len(pos[i_1]) * len(pos[i_2]) > 0:
                n_tot += len(pos[i_1]) * len(pos[i_2])
                input_queue.put(task)
        else:
            i = task
            if len(pos[i]) > 0:
                n_tot += len(pos[i])
                input_queue.put(i)

    if verbose:
        pbar = tqdm.tqdm(
            total=n_tot, bar_format='{l_bar}{bar}[{elapsed}<{remaining}]',
            smoothing=0)

    p_list = []
    for i in range(num_threads):
        p = Process(target=compute_tpcf, args=(
            mode, pos, tpcf, period, tpcf_args, tpcf_kwargs, input_queue,
            output_queue))
        p.start()
        p_list.append(p)

    while n < n_tot:
        try:
            task, xi = output_queue.get(False)

            if tpcf_matrix is None:
                if mode == 'auto':
                    tpcf_matrix = np.zeros(
                        (len(xi.ravel()), len(pos), len(pos)))
                else:
                    tpcf_matrix = np.zeros((len(xi.ravel()), len(pos)))

            if mode == 'auto':
                i_1, i_2 = task
                tpcf_matrix[:, i_1, i_2] += xi.ravel()
                tpcf_matrix[:, i_2, i_1] = tpcf_matrix[:, i_1, i_2]
                n += len(pos[i_1]) * len(pos[i_2])
                if verbose:
                    pbar.update(len(pos[i_1]) * len(pos[i_2]))
            else:
                i = task
                tpcf_matrix[:, i] += xi.ravel()
                n += len(pos[i])
                if verbose:
                    pbar.update(len(pos[i]))

            tpcf_shape = xi.shape

        except Empty:
            time.sleep(0.001)
            pass

    for p in p_list:
        p.join()

    if verbose:
        pbar.close()

    return tpcf_matrix, tpcf_shape


class TabCorr:

    def __init__(self):
        self.init = False

    @classmethod
    def tabulate(cls, halocat, tpcf, *tpcf_args,
                 mode='auto',
                 Num_ptcl_requirement=sim_defaults.Num_ptcl_requirement,
                 prim_haloprop_key=model_defaults.prim_haloprop_key,
                 prim_haloprop_bins=100,
                 sec_haloprop_key=model_defaults.sec_haloprop_key,
                 sec_haloprop_percentile_bins=None,
                 sats_per_prim_haloprop=3e-12, downsample=1.0,
                 verbose=False, redshift_space_distortions=True,
                 cens_prof_model=None, sats_prof_model=None, project_xyz=False,
                 cosmology_obs=None, num_threads=1, **tpcf_kwargs):
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

        sec_haloprop_percentile_bins : int, float or None, optional
            If an integer, it determines how many evenly spaced bins in the
            secondary halo property percentiles are going to be used. If a
            float between 0 and 1, it determines the split. If None is
            provided, no binning is applied.

        sats_per_prim_haloprop : float, optional
            Float determing how many satellites sample each halo. For each
            halo, the number is drawn from a Poisson distribution with an
            expectation value of ``sats_per_prim_haloprop`` times the primary
            halo property.

        downsample : float or function, optional
            Fraction between 0 and 1 used to downsample the total sample used
            to tabulate correlation functions. Values below unity can be used
            to reduce the computation time. It should not result in biases but
            the resulting correlation functions will be less accurate. If
            float, the same value is applied to all halos. If function, it
            should return the fraction is a function of the primary halo
            property.

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

        cosmology_obs : object, optional
            Instance of an astropy `~astropy.cosmology`. This can be used to
            correct coordinates in the simulation for the Alcock-Paczynski (AP)
            effect, i.e. a mismatch between the cosmology of the model
            (simulation) and the cosmology used to interpret observations. Note
            that the cosmology of the simulation is part of the halocat object.
            If None, no correction for the AP effect is applied. Also, a
            correction for the AP effect is only applied for auto-correlation
            functions.

        num_threads : int, optional
            How many threads to use for the tabulation.

        **tpcf_kwargs : dict, optional
                Keyword arguments passed to the ``tpcf`` function.

        Returns
        -------
        halotab : TabCorr
            Object containing all necessary information to calculate
            correlation functions for arbitrary galaxy models.
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
            prim_haloprop_bins = np.linspace(
                np.log10(np.amin(halos[prim_haloprop_key])) - 1e-3,
                np.log10(np.amax(halos[prim_haloprop_key])) + 1e-3,
                prim_haloprop_bins + 1)
        elif isinstance(prim_haloprop_bins, (list, np.ndarray)):
            pass
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

        n_h, prim_haloprop_bins, sec_haloprop_percentile_bins = (
            np.histogram2d(
                np.log10(halos[prim_haloprop_key]),
                halos[sec_haloprop_key + '_percentile'],
                bins=[prim_haloprop_bins, sec_haloprop_percentile_bins]))
        halotab.gal_type['n_h'] = n_h.ravel(order='F')

        grid = np.meshgrid(prim_haloprop_bins,
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

            pos_all = (return_xyz_formatted_array(
                x=gals[xyz[0]], y=gals[xyz[1]], z=gals[xyz[2]],
                velocity=gals['v'+xyz[2]] if redshift_space_distortions else 0,
                velocity_distortion_dimension='z', period=halocat.Lbox,
                redshift=halocat.redshift, cosmology=halocat.cosmology) *
                lbox_stretch)

            period = halocat.Lbox * lbox_stretch

            # Get a list of the positions of each sub-population.
            i_prim = np.digitize(np.log10(gals[prim_haloprop_key]),
                                 bins=prim_haloprop_bins, right=False) - 1
            mask = (i_prim < 0) | (i_prim >= len(prim_haloprop_bins))
            i_sec = np.digitize(
                gals[sec_haloprop_key + '_percentile'],
                bins=sec_haloprop_percentile_bins, right=False) - 1
            i_type = np.where(gals['gal_type'] == 'centrals', 0, 1)

            # Throw out those that don't fall into any bin.
            pos_all = pos_all[~mask]

            i = (i_prim +
                 i_sec * (len(prim_haloprop_bins) - 1) +
                 i_type * ((len(prim_haloprop_bins) - 1) *
                           (len(sec_haloprop_percentile_bins) - 1)))

            pos_all = pos_all[np.argsort(i)]
            counts = np.bincount(i, minlength=len(halotab.gal_type))

            assert len(counts) == len(halotab.gal_type)

            pos_bin = []
            for i in range(len(halotab.gal_type)):

                pos = pos_all[np.sum(counts[:i]):np.sum(counts[:i+1]), :]
                if halotab.gal_type['gal_type'][i] == 'centrals':
                    # Make sure the number of halos are consistent.
                    try:
                        assert len(pos) == int(halotab.gal_type['n_h'][i])
                    except AssertionError:
                        raise RuntimeError('There was an internal error in ' +
                                           'TabCorr. If possible, please ' +
                                           'report this bug in the TabCorr ' +
                                           'GitHub repository.')
                else:
                    if len(pos) == 0 and halotab.gal_type['n_h'][i] != 0:
                        raise RuntimeError(
                            'There was at least one bin without satellite ' +
                            'tracers. Increase sats_per_prim_haloprop.')

                if len(pos) > 0:

                    if isinstance(downsample, float):
                        use = np.random.random(len(pos)) < downsample
                    else:
                        use = (
                            np.random.random(len(pos)) <
                            downsample(halotab.gal_type['prim_haloprop'][i]))

                    # If the down-sampling reduced the number of tracers to at
                    # or below one, force at least 2 tracers to not bias the
                    # clustering estimates.
                    if np.sum(use) <= 1 and len(pos) > 1:
                        use = np.zeros(len(pos), dtype=bool)
                        use[np.random.choice(len(pos), size=2)] = True

                    pos = pos[use]

                pos_bin.append(pos)

            if xyz == 'xyz':
                tpcf_matrix, tpcf_shape = compute_tpcf_matrix(
                    mode, pos_bin, tpcf, period, tpcf_args, tpcf_kwargs,
                    num_threads=num_threads, verbose=verbose)

            if not project_xyz or mode == 'cross':
                break
            elif xyz != 'xyz':
                tpcf_matrix += compute_tpcf_matrix(
                    mode, pos_bin, tpcf, period, tpcf_args, tpcf_kwargs,
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
            However, arguments that are numpy arrays with more entries than
            max_args_size will be omitted.

        matrix_dtype : type
            The dtype used to write the correlation matrix to disk. You can use
            this to save space at the expense of predicsion.
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
            prim_haloprop=self.gal_type['prim_haloprop'].data[mask],
            sec_haloprop_percentile=(
                self.gal_type['sec_haloprop_percentile'].data[mask]),
            **occ_kwargs)
        mean_occupation[~mask] = model.mean_occupation_satellites(
            prim_haloprop=self.gal_type['prim_haloprop'].data[~mask],
            sec_haloprop_percentile=(
                self.gal_type['sec_haloprop_percentile'].data[~mask]),
            **occ_kwargs)
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
                xi = np.einsum('ij, j', self.tpcf_matrix, ngal_sq) / np.sum(ngal_sq)
            elif self.attrs['mode'] == 'cross':
                xi = np.einsum('ij, j', self.tpcf_matrix, ngal) / np.sum(ngal)
            return np.sum(ngal), xi
        else:

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
                    np.ma.masked_array(self.x, mask=(self.x <= x_model)))]
            else:
                if not extrapolate:
                    raise RuntimeError('The parameters of the model are ' +
                                       'outside of the interpolation range ' +
                                       'and extrapolation is turned off.')
                else:
                    simplex = np.argsort(np.abs(x_model - self.x))[:2]

            w1 = (self.x[simplex[1]] - x_model[0]) / (
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


def symmetric_matrix_to_array(matrix, check_symmetry=True):

    if check_symmetry:
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
