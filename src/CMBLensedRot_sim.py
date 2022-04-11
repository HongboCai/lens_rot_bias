"""This script generate CMB simulations with rotational field included"""

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
    'A_cb': 1E-7, 
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
                    
# define a geometry to work with
oshape, owcs = enmap.fullsky_geometry(args.pix_size*utils.arcmin)

ls, claa = np.loadtxt(args.alpha_ps, usecols=(0,1), unpack=True)

isim = args.sim_num

logging.info(f'doing sim {isim}, calling lensing.rand_map')
cmb_seed = (isim, 0, 0, 0)
alpha_seed = (isim, 0, 4, 0)
    
# load alpha alm and project to map space
print("Loading alpha alm")

alpha_alm = curvedsky.rand_alm(claa, lmax=args.lmax, seed=alpha_seed)                    
print("Generating alpha map")
alpha_map = curvedsky.alm2map(alpha_alm, enmap.zeros(oshape, owcs), spin=0)
print("writing alpha_fullsky_alm.fits")
hp.write_alm(cmb_dir + f"/alpha_fullsky_alm_{isim:03d}.fits", np.complex64(alpha_alm), overwrite=True)
del alpha_alm
    
# actually load cmb map alm
print("Loading cmb alm")
teb_alm = hp.read_alm(cmb_dir + f"/CMBLensed_fullsky_alm_{isim:03d}.fits", hdu=(1,2,3))                    

# convert to map space
print("Generating cmb map")
tqu_map = curvedsky.alm2map(teb_alm, enmap.zeros((3,)+oshape, owcs), spin=[0,2])
# del teb_alm

# rotate polarization
print("Rotating polarization by alpha")
rot_tqu_map = enmap.rotate_pol(tqu_map, alpha_map)
# pdb.set_trace()
    
# convert to alm
print("Converting map to alm")
rot_teb_alm = curvedsky.map2alm(rot_tqu_map, spin=[0,2], lmax=args.lmax)

del tqu_map

# write lensed rotated CMB alm
print("writing CMBLensedRot_fullsky_alm.fits")
hp.write_alm(cmb_dir + f"/CMBLensedRot_fullsky_alm_{isim:03d}.fits", np.complex128(rot_teb_alm), overwrite=True)



