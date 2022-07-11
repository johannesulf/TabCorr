"""Module for interpolation between TabCorr instances."""

import numpy as np
from scipy.spatial import Delaunay


class Interpolator:
    """Class for interpolation of multiple TabCorr instances."""

    def __init__(self, tabcorr_list, param_dict_table, spline=True):
        """Initialize an interpolation of multiple TabCorr instances.

        Parameters
        ----------
        tabcorr_list : list or numpy.ndarray
            TabCorr instances used to interpolate.
        param_dict_table : astropy.table.Table
            Table containing the keywords and values corresponding to each
            instance in the TabCorr list. Must have the same length and
            ordering as `tabcorr_list`.
        spline : bool, optional
            For multi-dimensional interpolation, whether the interpolation
            is performed over a regular grid using spline interpolation or
            over an irregular grid using linear barycentric interpolation.
            Spline interpolation is more accurate but slower and the TabCorr
            instances need to be arranged on a grid. Default is True.

        Raises
        ------
        ValueError
            If `spline` is True and `param_dict_table` does not describe a
            grid.

        """
        if len(tabcorr_list) != len(param_dict_table):
            raise ValueError("The number of TabCorr instances does not match" +
                             " the number of entries in 'param_dict_table'.")

        self.tabcorr_list = tabcorr_list
        self.keys = param_dict_table.colnames
        self.param_dict_table = param_dict_table.copy()
        self.spline = spline or len(self.keys) == 1

        if self.spline:
            self.xp = []
            self.a = []
            for key in self.keys:
                self.xp.append(np.sort(np.unique(param_dict_table[key])))
                self.a.append(spline_interpolation_matrix(self.xp[-1]))

            try:
                # Check that the table has the right length to describe the
                # grid.
                assert (np.prod([len(xp) for xp in self.xp]) ==
                        len(self.param_dict_table))
                # Check that no combination of values in the table appears
                # twice.
                assert np.all(
                    np.unique(np.stack(self.param_dict_table),
                              return_counts=True)[1] == 1)
            except AssertionError:
                raise ValueError(
                    "The 'param_dict_table' does not describe a grid.")

            self.param_dict_table['tabcorr_index'] = np.arange(len(
                self.param_dict_table))
            self.param_dict_table.sort(self.keys)

        else:

            if len(self.param_dict_table) <= len(self.keys):
                raise ValueError(
                    'The number of TabCorr instances provided must be ' +
                    'larger than the number of dimensions.')

            self.xp = np.zeros((len(self.param_dict_table), len(self.keys)))
            for i, key in enumerate(self.keys):
                self.xp[:, i] = self.param_dict_table[key].data
            self.delaunay = Delaunay(self.xp)

        # Determine unique halo tables such that we can save computation time
        # if halo tables are repeated.
        all_gal_type = [np.array(tabcorr.gal_type.as_array().tolist()).ravel()
                        for tabcorr in tabcorr_list]
        unique = np.unique(
            all_gal_type, axis=0, return_index=True, return_inverse=True)
        self.unique_gal_type_index = unique[1]
        self.unique_gal_type_inverse = unique[2]

    def predict(self, model, n_gauss_prim=10, extrapolate=False, **occ_kwargs):
        """Interpolate the predictions from multiple TabCorr instances.

        The values of parameters to interpolate should be in the parameter
        dictionary of the model.

        Parameters
        ----------
        model : HodModelFactory
            Instance of ``halotools.empirical_models.HodModelFactory``
            describing the model for which predictions are made.
        n_gauss_prim : int, optional
            The number of points used in the Gaussian quadrature to calculate
            the mean occupation averaged over the primary halo property in each
            halo bin. Default is 10.
        extrapolate : bool, optional
            Whether to allow extrapolation beyond points sampled by the input
            TabCorr instances. Default is False.
        **occ_kwargs : dict, optional
            Keyword arguments passed to the ``mean_occupation`` functions of
            the model.

        Returns
        -------
        ngal : float
            Galaxy number density.
        xi : numpy.ndarray
            Correlation function values.

        Raises
        ------
        ValueError
            If `extrapolate` is set to True and values are outside the
            interpolation range.

        """
        x_model = np.empty(len(self.keys))
        for i, key in enumerate(self.keys):
            try:
                x_model[i] = model.param_dict[key]
            except KeyError:
                raise ValueError(
                    'The key {} is not present in the parameter '.format(key) +
                    'dictionary of the model.')

        if self.spline:

            # Calculate the mean occupation numbers, avoiding to calculate
            # those repeatedly for identical halo tables.
            mean_occupation = [self.tabcorr_list[i].mean_occupation(
                model, n_gauss_prim=n_gauss_prim, **occ_kwargs) for i in
                self.unique_gal_type_index]

            for i in range(len(self.param_dict_table)):
                k = self.param_dict_table['tabcorr_index'][i]
                tabcorr = self.tabcorr_list[k]
                ngal_i, xi_i = tabcorr.predict(
                    mean_occupation[self.unique_gal_type_inverse[k]],
                    n_gauss_prim=n_gauss_prim, **occ_kwargs)
                if i == 0:
                    ngal = np.zeros(np.prod([len(xp) for xp in self.xp]))
                    xi = np.zeros([np.prod([len(xp) for xp in self.xp])] +
                                  list(xi_i.shape))
                ngal[i] = ngal_i
                xi[i] = xi_i
            ngal = ngal.reshape([len(xp) for xp in self.xp])
            xi = xi.reshape([len(xp) for xp in self.xp] + list(xi_i.shape))
            return (spline_interpolate(x_model, self.xp, self.a, ngal,
                                       extrapolate=extrapolate),
                    spline_interpolate(x_model, self.xp, self.a, xi,
                                       extrapolate=extrapolate))

        i_simplex = self.delaunay.find_simplex(x_model)

        if i_simplex == -1:
            if not extrapolate:
                raise ValueError(
                    'The x-coordinates are outside of the interpolation ' +
                    'range and extrapolation is turned off.')
            x_cm = np.mean(self.xp[self.delaunay.simplices], axis=1)
            i_simplex = np.argmin(np.sum((x_model - x_cm)**2, axis=1))

        simplex = self.delaunay.simplices[i_simplex]
        b = self.delaunay.transform[i_simplex, :-1].dot(
            x_model - self.delaunay.transform[i_simplex, -1])
        w = np.append(b, 1 - np.sum(b))

        for i, k in enumerate(simplex):
            ngal_i, xi_i = self.tabcorr_list[k].predict(
                model, **occ_kwargs)
            if i == 0:
                ngal = ngal_i * w[i]
                xi = xi_i * w[i]
            else:
                ngal += ngal_i * w[i]
                xi += xi_i * w[i]

        return ngal, xi


def spline_interpolation_matrix(xp):
    """Calculate a matrix for quick cubic not-a-knot spline interpolation.

    Parameters
    ----------
    xp : numpy.ndarray
        Abscissa for which to perform spline interpolation.

    Returns
    -------
    a : numpy.ndarray
        Matrix `a` such that ``np.einsum('ij,j,i', a[i], y, x0**np.arange(4))``
        is the value `i`-th spline evalualated at `x0`.

    Raises
    ------
    ValueError
        If `xp` does not have at least 4 entries.

    """
    if len(xp) < 4:
        raise ValueError('Cannot perform spline interpolation with less than' +
                         ' 4 values.')

    n = len(xp) - 1
    m = np.zeros((4 * n, 4 * n))

    # Ensure spline goes through y-values.
    for i in range(n):
        m[i][i*4:(i+1)*4] = xp[i]**np.arange(4)
        m[i+n][i*4:(i+1)*4] = xp[i+1]**np.arange(4)

    # Ensure continuity first and second derivative at the x-values.
    for i in range(n - 1):
        m[i+2*n][i*4+1:(i+1)*4] = np.array([1, 2, 3]) * xp[i+1]**np.arange(3)
        m[i+2*n][(i+1)*4+1:(i+2)*4] = -(
            np.array([1, 2, 3]) * xp[i+1]**np.arange(3))
        m[i+3*n-1][i*4+2:(i+1)*4] = np.array([2, 6]) * xp[i+1]**np.arange(2)
        m[i+3*n-1][(i+1)*4+2:(i+2)*4] = -(
            np.array([2, 6]) * xp[i+1]**np.arange(2))

    # Ensure continuity of third derivative in first and last x-values.
    m[-1][3] = 6 * xp[1]
    m[-1][7] = -6 * xp[1]
    m[-2][-5] = 6 * xp[-2]
    m[-2][-1] = -6 * xp[-2]

    # Invert matrix and determine matrix for y-values.
    m = np.linalg.inv(m)
    a = np.zeros((4 * n, len(xp)))
    a[:, :-1] = m[:, :n]
    a[:, 1:] += m[:, n:2*n]

    return a.reshape((n, 4, len(xp)))


def spline_interpolate(x, xp, a, yp, extrapolate=False):
    """Interpolate one or more possible multi-dimensional functions.

    This function can be used to interpolate multiple functions simultaneously
    as long as the x-coordinates of each interpolation are the same.

    Parameters
    ----------
    x : float or numpy.ndarray
        The single x-coordinate at which to evaluate the interpolated values.
    xp : numpy.ndarray or list of numpy.ndarray
        The x-coordinates of the data points. If multi-dimensional
        interpolation is performed, `x` and `xp` must have the same length.
    a : numpy.ndarray or list of numpy.ndarray
        The spline interpolation matrix or matrices calculated with
        ``spline_interpolation_matrix``. For multi-dimensional interpolation,
        must have the same length and order as `xp`.
    yp : numpy.ndarray
        The y-coordinates of the data points. If it has more dimensions than
        `x`, the interpolation is performed along the first ``len(x)`` axes.
    extrapolate : bool, optional
        Whether to allow extrapolation beyond points sampled by x. If set to
        false, attempting to extrapolate will result in a RuntimeError. Default
        is False.

    Returns
    -------
    yp : float or numpy.ndarray
        Interpolated value(s) with shape ``y.shape[len(x):]``.

    Raises
    ------
    ValueError
        If `x` falls outside the interpolation range and `extrapolate` is
        False.

    """
    if not isinstance(xp, list):
        xp = [xp]
    if not isinstance(a, list):
        a = [a]
    x = np.atleast_1d(x)

    for xi, ai, xpi in zip(x, a, xp):
        i_spline = np.digitize(xi, xpi) - 1
        if xi == xpi[-1]:
            i_spline = len(xpi) - 2
        if i_spline < 0 or i_spline >= len(xpi) - 1:
            if not extrapolate:
                raise ValueError(
                    'The x-coordinates are outside of the interpolation ' +
                    'range and extrapolation is turned off.')
            else:
                i_spline = min(max(i_spline, 0), len(xpi) - 2)
        yp = np.einsum('ij,j...,i', ai[i_spline], yp, xi**np.arange(4))

    return yp
