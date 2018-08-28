# TabCorr - Tabulated Correlation functions for halotools

This Python module provides extremely efficient and precise calculations of galaxy correlation functions in halotools using tabulated values. It is specifically intended for Markov chain monte carlo (MCMC) exploration of the galaxy-halo connection. It implements the method described in Zheng et al. (2016, http://adsabs.harvard.edu/abs/2016MNRAS.458.4015Z) of tabulating correlation functions that only need to be convolved with the mean halo occupation to obtain the full correlation function of galaxies.

---

### Prerequisites

The following python packages (and their prerequisites) are required for running this module.

* h5py
* numpy
* astropy
* halotools

This module has been tested with Python 3.x.

---

### Usage

The following code demonstrates the basic usage of TabCorr.

```
import numpy as np
from halotools.mock_observables import wp
from halotools.sim_manager import CachedHaloCatalog
from halotools.empirical_models import PrebuiltHodModelFactory
from tabcorr import TabCorr

# First, we tabulate the correlation functions in the halo catalog.
rp_bins = np.logspace(-1, 1, 20)
halocat = CachedHaloCatalog(simname='bolplanck')
halotab = TabCorr.tabulate(halocat, wp, rp_bins, pi_max=40)

# We can save the result for later use.
halotab.write('bolplanck.hdf5')

# We could read it in like this. Thus, we can skip the previous steps in the
# future.
halotab = TabCorr.read('bolplanck.hdf5')

# Now, we're ready to calculate correlation functions for a specific model.
model = PrebuiltHodModelFactory('zheng07')
ngal, wp = halotab.predict(model)
rp_ave = 0.5 * (rp_bins[1:] + rp_bins[:-1])

print("total number density: %.3e (Mpc/h)^-3" % np.sum(ngal))
print("projected correlation function:")
for i, rp in enumerate(rp_ave):
    print("%.3f Mpc/h: \t %.3e Mpc/h" % (rp, wp[i]))
```

It will generate the following output.
```
total number density: 5.664e-03 (Mpc/h)^-3
projected correlation function:
0.114 Mpc/h:     5.316e+02 Mpc/h
0.145 Mpc/h:     4.104e+02 Mpc/h
0.185 Mpc/h:     3.342e+02 Mpc/h
0.235 Mpc/h:     2.669e+02 Mpc/h
0.300 Mpc/h:     2.033e+02 Mpc/h
0.382 Mpc/h:     1.595e+02 Mpc/h
0.487 Mpc/h:     1.290e+02 Mpc/h
0.620 Mpc/h:     9.668e+01 Mpc/h
0.791 Mpc/h:     8.133e+01 Mpc/h
1.007 Mpc/h:     6.928e+01 Mpc/h
1.284 Mpc/h:     5.505e+01 Mpc/h
1.636 Mpc/h:     4.519e+01 Mpc/h
2.084 Mpc/h:     3.948e+01 Mpc/h
2.656 Mpc/h:     3.127e+01 Mpc/h
3.385 Mpc/h:     2.708e+01 Mpc/h
4.313 Mpc/h:     2.319e+01 Mpc/h
5.496 Mpc/h:     2.003e+01 Mpc/h
7.003 Mpc/h:     1.612e+01 Mpc/h
8.924 Mpc/h:     1.407e+01 Mpc/h
```

---

### To-do list

* Currently, TabCorr only works for autocorrelation functions. It should be
straightforward to also include cross-correlation functions, particularly
``delta_sigma``.
* TabCorr works for HOD and dHOD models. Support for SHAM models should be
easy to implement.
* Add a function that checks that the model and the tabulated halo catalog
are compatible.
* The phase-space distributions are hard-coded in right now. Specifically,
satellites use ``NFWPhaseSpace``. It should be straightforward to include
arbitrary phase-space models for centrals and satellites.
* Add option to project the halo catalog onto the x, y and z-axis to increase
precision.

---

### Author

Johannes Ulf Lange
