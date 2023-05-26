"""Generate CMBRot_fullsky_alms"""

import os, sys
from pixell import enmap, utils, lensing, aberration
from pixell import powspec, curvedsky
import numpy as np, healpy as hp, logging, os, os.path as op
import argparse
import pdb
import healpy as hp

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

parser = argparse.ArgumentParser()
parser.add_argument('--sim_num', type=int, help='the number of simulation')
parser.add_argument('--alpha_num', type=int, help='the number of alpha map')
parser.add_argument("--odir",       type=str, default=defaults['odir'], help="Output directory")
parser.add_argument("--lmax",       type=int, default=defaults['lmax'],       help="Max multipole for lensing")
parser.add_argument("--lmax-write", type=int, default=defaults['lmax_write'], help="Max multipole to write")
parser.add_argument("--pix-size",   type=float, default=defaults['pix_size'], help="Pixel width in arcmin")
                    
args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)
cmb_dir = args.odir
                    
# define a geometry to work with
oshape, owcs = enmap.fullsky_geometry(args.pix_size*utils.arcmin)

isim = args.sim_num
alpha_num = args.alpha_num

# read in alpha alm and project to map space
print("Reading in alpha_alm")
alpha_alm = hp.read_alm(cmb_dir + f"/alpha_fullsky_alm_{defaults['A_cb']}_{alpha_num:03d}.fits")
print("Generating alpha map")
alpha_map = curvedsky.alm2map(alpha_alm, enmap.zeros(oshape, owcs), spin=0)

# read in cmb alm and project to map space
print("Reading in cmb_alm")
teb_alm = hp.read_alm(cmb_dir + f"/CMB_fullsky_alm_{isim:03d}.fits", hdu=(1,2,3))

# convert to map space
print("Generating cmb map")
tqu_map = curvedsky.alm2map(teb_alm, enmap.zeros((3,)+oshape, owcs), spin=[0,2])

# rotate polarization
print("Rotating polarization by alpha")
rot_tqu_map = enmap.rotate_pol(tqu_map, alpha_map)

# convert to alm
print("Converting map to alm")
rot_teb_alm = curvedsky.map2alm(rot_tqu_map, spin=[0,2], lmax=args.lmax)

del tqu_map

# write rotated CMB alm
print("writing CMBRot_fullsky_alm.fits")
hp.write_alm(cmb_dir + f"/CMBRot_fullsky_alm_{defaults['A_cb']}_{isim:03d}.fits", np.complex128(rot_teb_alm), overwrite=True)


