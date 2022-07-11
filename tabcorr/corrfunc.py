"""Module providing wrappers around Corrfunc functions."""

import numpy as np


def wp(sample1, rp_bins, pi_max, sample2=None, period=None, do_auto=True,
       do_cross=False):
    """Corrfunc implementation of `halotools.mock_observables.wp`.

    This function provides a wrapper around `corrfunc.theory.DDrppi` such that
    when used by TabCorr behaves the same as `halotools.mock_observables.wp`.
    However, not all functionality of `halotools.mock_observables.wp` is
    implemented.

    Parameters
    ----------
    sample1 : numpy.ndarray
        Numpy array containing the positions of points.
    rp_bins : numpy.ndarray
        Boundaries defining the radial bins in which pairs are counted.
    pi_max : float
        Maximum line-of-sight distance.
    sample2 : numpy.ndarray, optional
        Numpy array containing the positions of a second sample of points.
        Default is None.
    period : float or numpy.ndarray, optional
        If a numpy array the periodic boundary conditions in each
        dimension. If a single scalar,  period is assumed to be the same in all
        dimensions. Default is None.
    do_auto : bool, optional
        Whether the calculate the auto-correlation function for `sample1`. If
        True, returns the auto-correlation function for `sample1`, regardless
        of whether `sample2` is given. If False, will return the
        cross-correlation between `sample1` and `sample2`. Default is True.
    do_cross : bool, optional
        Whether the calculate the auto-correlation function for `sample1` and
        `sample2`. Default is False.

    Returns
    -------
    wp : numpy.ndarray
        The two-point correlation function.

    Raises
    ------
    RuntimeError
        If Corrfunc is not installed.
    ValueError
        If `do_auto` and `do_cross` have the same value.

    """
    try:
        from Corrfunc.theory import DDrppi
    except ImportError:
        raise RuntimeError('Corrfunc needs to be installed to use wrappers.')

    if (do_auto and do_cross) or (not do_auto and not do_cross):
        raise ValueError("'do_auto' and 'do_cross' cannot both be True or " +
                         "False.")

    if isinstance(period, float) or isinstance(period, int):
        period = (period, period, period)

    if isinstance(period, np.ndarray):
        period = tuple(period)

    if do_auto:
        r = DDrppi(1, 1, pi_max, rp_bins, sample1[:, 0], sample1[:, 1],
                   sample1[:, 2], periodic=True, boxsize=period,
                   xbin_refine_factor=1, ybin_refine_factor=1,
                   zbin_refine_factor=1)
        n_exp = (len(sample1) * len(sample1) / np.prod(period) * np.pi *
                 np.diff(rp_bins**2) * 2 * pi_max)

    else:
        r = DDrppi(0, 1, pi_max, rp_bins, sample1[:, 0], sample1[:, 1],
                   sample1[:, 2], periodic=True, boxsize=period,
                   X2=sample2[:, 0], Y2=sample2[:, 1], Z2=sample2[:, 2],
                   xbin_refine_factor=1, ybin_refine_factor=1,
                   zbin_refine_factor=1)
        n_exp = (len(sample1) * len(sample2) / np.prod(period) * np.pi *
                 np.diff(rp_bins**2) * 2 * pi_max)

    npairs = r['npairs']
    npairs = np.array([np.sum(n) for n in np.split(npairs, len(rp_bins) - 1)])

    return (npairs / n_exp - 1) * 2 * pi_max


def s_mu_tpcf(sample1, s_bins, mu_bins, sample2=None, period=None,
              do_auto=True, do_cross=False):
    """Corrfunc implementation of `halotools.mock_observables.s_mu_tpcf`.

    This function provides a wrapper around `corrfunc.theory.DDsmu` such that
    when used by TabCorr behaves the same as
    `halotools.mock_observables.s_mu_tpcf`. However, not all functionality of
    `halotools.mock_observables.s_mu_tpcf` is implemented.

    Parameters
    ----------
    sample1 : numpy.ndarray
        Numpy array containing the positions of points.
    s_bins : numpy.ndarray
        Boundaries defining the distance bins in which pairs are counted.
    mu_bins : numpy.ndarray
        Boundaries defining the angular bins in which pairs are counted.
    sample2 : numpy.ndarray, optional
        Numpy array containing the positions of a second sample of points.
        Default is None.
    period : float or numpy.ndarray, optional
        If a numpy array the periodic boundary conditions in each
        dimension. If a single scalar,  period is assumed to be the same in all
        dimensions. Default is None.
    do_auto : bool, optional
        Whether the calculate the auto-correlation function for `sample1`. If
        True, returns the auto-correlation function for `sample1`, regardless
        of whether `sample2` is given. If False, will return the
        cross-correlation between `sample1` and `sample2`. Default is True.
    do_cross : bool, optional
        Whether the calculate the auto-correlation function for `sample1` and
        `sample2`. Default is False.

    Returns
    -------
    xi : numpy.ndarray
        The two-point correlation function.

    Raises
    ------
    RuntimeError
        If Corrfunc is not installed.
    ValueError
        If `do_auto` and `do_cross` have the same value or if `mu_bins` are not
        uniform bins from 0 to 1.

    """
    try:
        from Corrfunc.theory import DDsmu
    except ImportError:
        raise RuntimeError('Corrfunc needs to be installed to use wrappers.')

    if (do_auto and do_cross) or (not do_auto and not do_cross):
        raise ValueError("'do_auto' and 'do_cross' cannot both be True or " +
                         "False.")

    try:
        assert np.all(np.isclose(mu_bins, np.linspace(0, 1, len(mu_bins))))
    except AssertionError:
        raise ValueError('Bins in mu must be uniform from 0 to 1.')

    if isinstance(period, float) or isinstance(period, int):
        period = (period, period, period)

    if isinstance(period, np.ndarray):
        period = tuple(period)

    if do_auto:
        r = DDsmu(1, 1, s_bins, 1, len(mu_bins) - 1, sample1[:, 0],
                  sample1[:, 1], sample1[:, 2], periodic=True,
                  boxsize=period, xbin_refine_factor=1,
                  ybin_refine_factor=1, zbin_refine_factor=1)
        n_exp = (len(sample1) * len(sample1) / np.prod(period) * 4 *
                 np.pi / 3 * np.diff(s_bins**3) / (len(mu_bins) - 1))

    else:
        r = DDsmu(0, 1, s_bins, 1, len(mu_bins) - 1, sample1[:, 0],
                  sample1[:, 1], sample1[:, 2], periodic=True,
                  boxsize=period, X2=sample2[:, 0],
                  Y2=sample2[:, 1], Z2=sample2[:, 2], xbin_refine_factor=1,
                  ybin_refine_factor=1, zbin_refine_factor=1)
        n_exp = (len(sample1) * len(sample2) / np.prod(period) * 4 *
                 np.pi / 3 * np.diff(s_bins**3) / (len(mu_bins) - 1))

    return (r['npairs'].reshape((len(s_bins) - 1, len(mu_bins) - 1)) /
            n_exp[:, np.newaxis] - 1)
