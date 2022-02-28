"""Generate CMBLensed_fullsky_alms"""

"""Modified from actsims/bin/signalGen, in fact much simplified
since I don't need a fully MPI-based sim generator"""

import os, sys
from pixell import enmap, utils, lensing, aberration
from pixell import powspec, curvedsky
import numpy as np, healpy as hp, logging, os, os.path as op
import argparse

# add the parent dir in the python path
sys.path.append(os.path.dirname(os.getcwd()))
import param as p

defaults = {
    'odir': '../simMaps',
    'lmax': p.lmax,
    'lmax_write': p.lmax_write,
    'pix_size': p.px_arcmin,
    'phi_ps': '../inputPs/cosmo2017_10K_acc3_lenspotentialCls.dat',
}

parser = argparse.ArgumentParser()
# Parse command line
parser = argparse.ArgumentParser(description='Generate lensed CMB')
parser.add_argument("--sim_num", type=int, help="the number of simulation")
parser.add_argument("--odir",       type=str, default=defaults['odir'], help="Output directory")
parser.add_argument("--lmax",       type=int, default=defaults['lmax'],       help="Max multipole for lensing")
parser.add_argument("--lmax-write", type=int, default=defaults['lmax_write'], help="Max multipole to write")
parser.add_argument("--pix-size",   type=float, default=defaults['pix_size'], help="Pixel width in arcmin")
parser.add_argument("--phi_ps",         type=str, default=defaults['phi_ps'], help="Input phi cl file")

args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)
cmb_dir = args.odir

logging.basicConfig(filename=f'{cmb_dir}/log.txt', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',filemode='w')

shape, wcs = enmap.fullsky_geometry(args.pix_size*utils.arcmin)
phi_ps = powspec.read_camb_full_lens(args.phi_ps)

#make phi totally uncorrelated with both T and E.  This is necessary due to the way that separate phi and CMB seeds were put forward in an update to the pixell library around mid-Nov 2018
phi_ps[0, 1:, :] = 0.
phi_ps[1:, 0, :] = 0.

# get clpp and claa
ls, clpp = np.loadtxt(args.phi_ps, usecols=(0,5), unpack=True)
factor = 0.5 * ls**2


isim = args.sim_num

logging.info(f'doing sim {isim}, calling lensing.rand_map')
cmb_seed = (isim, 0, 0, 0)
phi_seed = (isim, 0, 2, 0)

# generate and write lensed CMB alm
l_tqu_map, = lensing.rand_map((3,)+shape, wcs, phi_ps,
                            lmax=args.lmax,
                            output="l",
                            phi_seed=phi_seed,
                            seed=cmb_seed,
                            verbose=True)
    
logging.info('calling curvedsky.map2alm')
alm = curvedsky.map2alm(l_tqu_map, lmax=args.lmax_write, spin=[0,2])

filename = cmb_dir + f"/CMBLensed_fullsky_alm_{isim:03d}.fits"
logging.info(f'writing to disk, filename is {filename}')
hp.write_alm(filename, np.complex64(alm), overwrite=True)
del alm
    
# generate phi and kappa alm, write kappa_alm
phi_alm = curvedsky.rand_alm(clpp, lmax=args.lmax, seed=phi_seed)
kappa_alm = hp.almxfl(phi_alm, factor)
hp.write_alm(cmb_dir + f'/kappa_fullsky_alm_{isim:03d}.fits', kappa_alm, overwrite=True)
del phi_alm, kappa_alm


