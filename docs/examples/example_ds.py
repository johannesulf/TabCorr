import numpy as np
import matplotlib.pyplot as plt
from halotools.sim_manager import CachedHaloCatalog
from halotools.mock_observables import mean_delta_sigma
from halotools.mock_observables import return_xyz_formatted_array
from halotools.empirical_models import PrebuiltHodModelFactory
from tabcorr import TabCorr

# First, we tabulate the correlation functions in the halo catalog.
rp_bins = np.logspace(-1, 1, 20)

halocat = CachedHaloCatalog(simname='bolplanck')
ptcl = halocat.ptcl_table
pos_ptcl = return_xyz_formatted_array(ptcl['x'], ptcl['y'], ptcl['z'])
effective_particle_mass = halocat.particle_mass * 2048**3 / len(ptcl)
halotab = TabCorr.tabulate(
    halocat, mean_delta_sigma, pos_ptcl, effective_particle_mass, rp_bins,
    mode='cross', verbose=True, num_threads=4)

# We can save the result for later use.
halotab.write('bolplanck_ds.hdf5')

# We could read it in like this. Thus, we can skip the previous steps in the
# future.
halotab = TabCorr.read('bolplanck_ds.hdf5')

# Now, we're ready to calculate correlation functions for a specific model.
model = PrebuiltHodModelFactory('zheng07', threshold=-21)

rp_ave = 0.5 * (rp_bins[1:] + rp_bins[:-1])

ngal, ds = halotab.predict(model)
plt.plot(rp_ave, rp_ave * ds / 1e12, label='total')

ngal, ds = halotab.predict(model, separate_gal_type=True)
for key in ds.keys():
    plt.plot(rp_ave, rp_ave * ds[key] / 1e12, label=key, ls='--')

plt.xscale('log')
plt.xlabel(r'$r_p \ [h^{-1} \ \mathrm{Mpc}]$')
plt.ylabel(r'$r_p \times \Delta\Sigma \ [10^6 \, M_\odot / \mathrm{pc}]$')
plt.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig('ds_decomposition.png', dpi=300)
plt.close()
