"""Generate Alpha_fullsky_alms"""

import os, sys
from pixell import enmap, utils, lensing, aberration
from pixell import powspec, curvedsky
import numpy as np, healpy as hp, logging, os, os.path as op
import argparse
import pdb

# add the parent dir in the python path
sys.path.append(os.path.dirname(os.getcwd()))
import param as p

defaults = {
    'odir': '../Maps',
    'lmax': p.lmax,
    'lmax_write': p.lmax_write,
    'pix_size': p.px_arcmin,
    'A_cb': 1E-8, 
}

alpha_ps = f'../inputPs/claa_A%s.txt'%defaults['A_cb']

parser = argparse.ArgumentParser()
parser.add_argument('--sim_num', type=int, help='the number of simulation')
parser.add_argument("--odir",       type=str, default=defaults['odir'], help="Output directory")
parser.add_argument("--lmax",       type=int, default=defaults['lmax'],       help="Max multipole for lensing")
parser.add_argument("--lmax-write", type=int, default=defaults['lmax_write'], help="Max multipole to write")
parser.add_argument("--pix-size",   type=float, default=defaults['pix_size'], help="Pixel width in arcmin")
parser.add_argument("--alpha_ps",         type=str, default=alpha_ps, help="Input alpha file")

args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)
cmb_dir = args.odir

ls, claa = np.loadtxt(args.alpha_ps, usecols=(0,1), unpack=True)

isim = args.sim_num

logging.info(f'doing sim {isim}, calling lensing.rand_map')

alpha_seed = (isim, 0, 4, 0)

# load alpha alm and project to map space
print("Loading alpha alm")

alpha_alm = curvedsky.rand_alm(claa, lmax=args.lmax, seed=alpha_seed)
print("writing alpha_fullsky_alm.fits")
hp.write_alm(cmb_dir + f"/alpha_fullsky_alm_{defaults['A_cb']}_{isim:03d}.fits", np.complex64(alpha_alm), overwrite=True)
del alpha_alm


