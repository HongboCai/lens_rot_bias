"""Generate CMB_fullsky_alms"""

import os, sys
from pixell import enmap, utils, lensing, aberration
from pixell import powspec, curvedsky
import numpy as np, healpy as hp, logging, os, os.path as op
import argparse

# add the parent dir in the python path
sys.path.append(os.path.dirname(os.getcwd()))
import param as p
from math import pi

defaults = {
    'odir': p.repodir + 'Maps',
    'lmax': p.lmax,
    'lmax_write': p.lmax_write,
    'pix_size': p.px_arcmin
}

parser = argparse.ArgumentParser()
# Parse command line
parser = argparse.ArgumentParser(description='Generate lensed CMB')
parser.add_argument("--sim_num", type=int, help="the number of simulation")
parser.add_argument("--odir",       type=str, default=defaults['odir'], help="Output directory")
parser.add_argument("--lmax",       type=int, default=defaults['lmax'],       help="Max multipole for lensing")
parser.add_argument("--lmax-write", type=int, default=defaults['lmax_write'], help="Max multipole to write")
parser.add_argument("--pix-size",   type=float, default=defaults['pix_size'], help="Pixel width in arcmin")

args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)
cmb_dir = args.odir

shape, wcs = enmap.fullsky_geometry(args.pix_size*utils.arcmin)

isim = args.sim_num

# logging.info(f'doing sim {isim}, calling lensing.rand_map')
cmb_seed = (isim, 0, 0, 0)

# generate and write CMB_alm.fits
alm = curvedsky.rand_alm()


l_tqu_map, = lensing.rand_map((3,)+shape, wcs, phi_ps,
                            lmax=args.lmax,
                            output="l",
                            phi_seed=phi_seed,
                            seed=cmb_seed,
                            verbose=True)


