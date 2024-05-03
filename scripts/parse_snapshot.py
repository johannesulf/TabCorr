import argparse
import gc
import io
import numpy as np
import os
import requests
import struct

from abacusnbody.data.read_abacus import read_asdf
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from astropy import units as u
from astropy.cosmology import w0waCDM
from astropy.table import Table, vstack
from collections import namedtuple
from halotools.empirical_models import delta_vir
from pathlib import Path
from tabcorr import database
from tqdm import tqdm


ABACUS_SUMMIT_PATH_BASE = Path('/global/cfs/cdirs/desi/cosmosim/Abacus')


def read_gadget_snapshot(bstream, read_pos=False, read_vel=False,
                         read_id=False, read_mass=False, print_header=False,
                         single_type=-1, lgadget=False):
    """
    Read a Gadget-2 snapshot file.

    This is a modified version of the function readGadgetSnapshot by Yao-Yuan
    Mao licensed under the MIT License. (https://bitbucket.org/yymao/helpers)

    Parameters
    ----------
    bstream : io.BytesIO
        Binary stream of the gadget snapshot file.
    read_pos : bool, optional
        Whether to read the positions or not. Default is false.
    read_vel : bool, optional
        Whether to read the velocities or not. Default is false.
    read_id : bool, optional
        Whether to read the particle IDs or not. Default is false.
    read_mass : bool, optional
        Whether to read the masses or not. Default is false.
    print_header : bool, optional
        Whether to print out the header or not. Default is false.
    single_type : int, optional
        Set to -1 (default) to read in all particle types.
        Set to 0--5 to read in only the corresponding particle type.
    lgadget : bool, optional
        Set to True if the particle file comes from l-gadget.
        Default is false.

    Returns
    -------
    ret : tuple
        A tuple of the requested data.
        The first item in the returned tuple is always the header.
        The header is in the gadget_header namedtuple format.

    """
    gadget_header_fmt = '6I6dddii6Iiiddddii6Ii'

    gadget_header = namedtuple(
        'gadget_header', 'npart mass time redshift flag_sfr flag_feedback ' +
        'npartTotal flag_cooling num_files BoxSize Omega0 OmegaLambda ' +
        'HubbleParam flag_age flag_metals NallHW flag_entr_ics')

    blocks_to_read = (read_pos, read_vel, read_id, read_mass)
    ret = []

    bstream.seek(4, 1)
    h = list(struct.unpack(gadget_header_fmt,
             bstream.read(struct.calcsize(gadget_header_fmt))))
    if lgadget:
        h[30] = 0
        h[31] = h[18]
        h[18] = 0
        single_type = 1
    h = tuple(h)
    header = gadget_header._make(
        (h[0:6],) + (h[6:12],) + h[12:16] + (h[16:22],) + h[22:30] +
        (h[30:36],) + h[36:])
    if print_header:
        print(header)
    if not any(blocks_to_read):
        return header
    ret.append(header)
    bstream.seek(256 - struct.calcsize(gadget_header_fmt), 1)
    bstream.seek(4, 1)

    mass_npart = [0 if m else n for m, n in zip(header.mass, header.npart)]
    if single_type not in set(range(6)):
        single_type = -1

    for i, b in enumerate(blocks_to_read):
        fmt = np.dtype(np.float32)
        fmt_64 = np.dtype(np.float64)
        item_per_part = 1
        npart = header.npart

        if i < 2:
            item_per_part = 3
        elif i == 2:
            fmt = np.dtype(np.uint32)
            fmt_64 = np.dtype(np.uint64)
        elif i == 3:
            if sum(mass_npart) == 0:
                ret.append(np.array([], fmt))
                break
            npart = mass_npart

        size_check = struct.unpack('I', bstream.read(4))[0]

        block_item_size = item_per_part*sum(npart)
        if size_check != block_item_size*fmt.itemsize:
            fmt = fmt_64
        if size_check != block_item_size*fmt.itemsize:
            raise ValueError('Invalid block size in file!')
        size_per_part = item_per_part*fmt.itemsize
        #
        if not b:
            bstream.seek(sum(npart)*size_per_part, 1)
        else:
            if single_type > -1:
                bstream.seek(sum(npart[:single_type])*size_per_part, 1)
                npart_this = npart[single_type]
            else:
                npart_this = sum(npart)
            data = np.frombuffer(bstream.read(npart_this*size_per_part), fmt)
            if item_per_part > 1:
                data.shape = (npart_this, item_per_part)
            ret.append(data)
            if not any(blocks_to_read[i+1:]):
                break
            if single_type > -1:
                bstream.seek(sum(npart[single_type+1:])*size_per_part, 1)
        bstream.seek(4, 1)

    return tuple(ret)


def download_aemulus_alpha_halos(simulation, redshift):

    try:
        username = os.environ['AEMULUS_USERNAME']
        password = os.environ['AEMULUS_PASSWORD']
    except KeyError:
        raise RuntimeError("Set the AEMULUS_USERNAME and AEMULUS_PASSWORD " +
                           "environment variables.")

    scale_factor_snapshots = np.array([0.25, 0.333333, 0.5, 0.540541, 0.588235,
                                       0.645161, 0.714286, 0.8, 0.909091, 1.0])
    redshift_snapshots = 1 / scale_factor_snapshots - 1

    try:
        assert np.amin(np.abs(redshift_snapshots - redshift)) < 0.005
    except AssertionError:
        raise ValueError('No snapshot for redshift {:.2f}.'.format(redshift))

    snapnum = np.argmin(np.abs(redshift_snapshots - redshift))

    url = ("https://www.slac.stanford.edu/~jderose/aemulus/phase1/" +
           "{}/halos/m200b/outbgc2_{}.list".format(simulation, snapnum))

    r = requests.get(url, auth=requests.auth.HTTPBasicAuth(username, password))
    halos = Table.read(io.BytesIO(r.content), format='ascii', delimiter=' ')

    url = url.replace('outbgc2', 'out')
    r = requests.get(url, auth=requests.auth.HTTPBasicAuth(username, password))
    halos['halo_rs'] = Table.read(
        io.BytesIO(r.content), format='ascii', delimiter=' ')['col7'] / 1e3
    halos['R200b'] /= 1e3

    halos.rename_column('M200b', 'halo_m200m')
    halos.rename_column('R200b', 'halo_r200m')
    halos.rename_column('Vmax', 'halo_vmax')
    for coordinate in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
        halos.rename_column(coordinate.upper(), 'halo_{}'.format(coordinate))

    halos = halos[halos['Parent_ID'] == -1]

    halos.keep_columns([col for col in halos.colnames if col[:5] == 'halo_'])

    return halos


def download_aemulus_alpha_particles(simulation, redshift):

    try:
        username = os.environ['AEMULUS_USERNAME']
        password = os.environ['AEMULUS_PASSWORD']
    except KeyError:
        raise RuntimeError("Set the AEMULUS_USERNAME and AEMULUS_PASSWORD " +
                           "environment variables.")

    scale_factor_snapshots = np.array([0.25, 0.333333, 0.5, 0.540541, 0.588235,
                                       0.645161, 0.714286, 0.8, 0.909091, 1.0])
    redshift_snapshots = 1 / scale_factor_snapshots - 1

    try:
        assert np.amin(np.abs(redshift_snapshots - redshift)) < 0.005
    except AssertionError:
        raise ValueError('No snapshot for redshift {:.2f}.'.format(redshift))

    snapnum = np.argmin(np.abs(redshift_snapshots - redshift))

    ptcls = np.zeros((0, 3))

    for chunk in tqdm(range(512)):
        url = ("https://www.slac.stanford.edu/~jderose/aemulus/phase1/" +
               "{}/output/snapdir_{:03}/snapshot_{:03}.{}".format(
                   simulation, snapnum, snapnum, chunk))
        r = requests.get(
            url, auth=requests.auth.HTTPBasicAuth(username, password))
        ptcls_tmp = read_gadget_snapshot(
            io.BytesIO(r.content), read_pos=True)[1]
        ptcls_tmp = ptcls_tmp[np.random.random(len(ptcls_tmp)) < 0.01]
        ptcls = np.vstack([ptcls, ptcls_tmp])

    return Table([ptcls[:, 0], ptcls[:, 1], ptcls[:, 2]],
                 names=('x', 'y', 'z'))


def read_abacus_summit_halos(simulation, redshift):

    fields = ['x_L2com', 'v_L2com', 'N', 'rvcirc_max_com']
    halocat = CompaSOHaloCatalog(
        ABACUS_SUMMIT_PATH_BASE / 'AbacusSummit_{}'.format(simulation) /
        'halos' / 'z{:.3f}'.format(redshift), fields=fields)
    halocat.halos = halocat.halos[halocat.halos['N'] >= 300]
    halos = halocat.halos

    mdef = '{:.0f}m'.format(halocat.header['SODensityL1'])
    halos['halo_m{}'.format(mdef)] = (
        halos['N'] * halocat.header['ParticleMassHMsun'])
    halos.remove_column('N')

    halos['x_L2com'] += halocat.header['BoxSize'] / 2.0
    halos['halo_x'] = halos['x_L2com'][:, 0]
    halos['halo_y'] = halos['x_L2com'][:, 1]
    halos['halo_z'] = halos['x_L2com'][:, 2]
    halos.remove_column('x_L2com')

    halos['halo_vx'] = halos['v_L2com'][:, 0]
    halos['halo_vy'] = halos['v_L2com'][:, 1]
    halos['halo_vz'] = halos['v_L2com'][:, 2]
    halos.remove_column('v_L2com')

    cosmology = w0waCDM(H0=halocat.header['H0'], Om0=halocat.header['Omega_M'],
                        Ode0=halocat.header['Omega_DE'],
                        w0=halocat.header['w0'], wa=halocat.header['wa'])
    dvir = delta_vir(cosmology, redshift) * 200 / (18 * np.pi**2)
    rho_crit = (cosmology.critical_density(redshift) /
                (cosmology.H(0).value / 100)**2 / (1 + redshift)**3)
    halocat.halos['halo_r{}'.format(mdef)] = ((
        halocat.halos['halo_m{}'.format(mdef)] * u.M_sun / (
            4.0 / 3.0 * np.pi * rho_crit * dvir))**(1.0 / 3.0)).to(u.Mpc).value

    halos['rvcirc_max_com'] /= 2.16258
    halos.rename_column('rvcirc_max_com', 'halo_rs')

    return halos


def read_abacus_summit_particles(simulation, redshift):

    ptcls = []
    for ptcl_type in ['field', 'halo']:
        for i in range(34):
            path = (
                ABACUS_SUMMIT_PATH_BASE / 'AbacusSummit_{}'.format(
                    simulation) / 'halos' / 'z{:.3f}'.format(redshift) /
                '{}_rv_A'.format(ptcl_type) /
                '{}_rv_A_{:03d}.asdf'.format(ptcl_type, i))
            ptcls_tmp = read_asdf(path, load=['pos'])
            ptcls_tmp = ptcls_tmp[
                np.random.random(len(ptcls_tmp)) < 0.00025 / 0.03]
            ptcls.append(ptcls_tmp)
            gc.collect()

    ptcls = vstack(ptcls)

    path = (ABACUS_SUMMIT_PATH_BASE / 'AbacusSummit_{}'.format(
        simulation) / 'info' / 'abacus.par')
    with open(path) as fstream:
        line = fstream.readlines()[3]
        assert 'BoxSize' in line
        boxsize = float(line.split('=')[1])
    ptcls['x'] = ptcls['pos'][:, 0] + boxsize / 2.0
    ptcls['y'] = ptcls['pos'][:, 1] + boxsize / 2.0
    ptcls['z'] = ptcls['pos'][:, 2] + boxsize / 2.0
    ptcls.remove_column('pos')
    return ptcls


def main():

    parser = argparse.ArgumentParser(
        description='Download/read and reduce an Aemulus Alpha or Abacus' +
        'Summit simulation. For AbacusSummit, you need to run this script ' +
        'on NERSC.')
    parser.add_argument('suite', help='simulation suite',
                        choices=['AemulusAlpha', 'AbacusSummit'])
    parser.add_argument('redshift', help='simulation redshift', type=float)
    parser.add_argument('--cosmo', help='simulation cosmology, default is 0',
                        type=int, default=0)
    parser.add_argument('--phase', help='simulation phase, default is 0',
                        type=int, default=0)
    parser.add_argument('--config', default=None,
                        help='simulation configuration to assume')
    parser.add_argument('--particles', action='store_true',
                        help='whether to download particles instead of halos')

    args = parser.parse_args()

    name = database.simulation_name(
        args.suite, i_cosmo=args.cosmo, i_phase=args.phase, config=args.config)

    print('Parsing data for {} at z={:.2f}...'.format(name, args.redshift))

    path = database.simulation_snapshot_directory(
        args.suite, args.redshift, i_cosmo=args.cosmo, i_phase=args.phase,
        config=args.config)
    path.mkdir(parents=True, exist_ok=True)

    if not args.particles:
        subpath = 'halos'
        if args.suite == 'AemulusAlpha':
            data = download_aemulus_alpha_halos(name, args.redshift)
        else:
            data = read_abacus_summit_halos(name, args.redshift)
    else:
        subpath = 'particles'
        if args.suite == 'AemulusAlpha':
            data = download_aemulus_alpha_particles(name, args.redshift)
        else:
            data = read_abacus_summit_particles(name, args.redshift)

    print('Writing results to {}.'.format(path / 'snapshot.hdf5'))
    data.write(path / 'snapshot.hdf5', path=subpath, overwrite=True,
               append=True)
    print('Done!')


if __name__ == "__main__":
    main()
