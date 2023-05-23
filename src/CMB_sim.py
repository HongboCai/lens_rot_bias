"""Generate CMB_fullsky_alms"""

import os, sys
from pixell import enmap, utils, lensing, aberration
from pixell import powspec, curvedsky
import numpy as np, healpy as hp, logging, os, os.path as op
import argparse
from numpy import pi

# add the parent dir in the python path
sys.path.append(os.path.dirname(os.getcwd()))
import param as p
from math import pi

defaults = {
    'odir': p.repodir + 'Maps',
    'lmax': p.lmax,
    'lmax_write': p.lmax_write,
    'pix_size': p.px_arcmin,
    'input_ps': '../inputPs/cosmo2017_10K_acc3_lenspotentialCls.dat',
}

parser = argparse.ArgumentParser()
# Parse command line
parser = argparse.ArgumentParser(description='Generate lensed CMB')
parser.add_argument("--sim_num", type=int, help="the number of simulation")
parser.add_argument("--odir",       type=str, default=defaults['odir'], help="Output directory")
parser.add_argument("--lmax",       type=int, default=defaults['lmax'],       help="Max multipole for lensing")
parser.add_argument("--lmax-write", type=int, default=defaults['lmax_write'], help="Max multipole to write")
parser.add_argument("--pix-size",   type=float, default=defaults['pix_size'], help="Pixel width in arcmin")
parser.add_argument("--input_ps",         type=str, default=defaults['input_ps'], help="Input unlensed CMB and CMB lensing cl file")

args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)
cmb_dir = args.odir

ls = np.arange(args.lmax+1)
fac = ls*(ls+1)/(2*pi)
fac[0] = 1

shape, wcs = enmap.fullsky_geometry(args.pix_size*utils.arcmin)
input_ps = np.loadtxt(args.input_ps)
cltt = np.concatenate(([0,0], input_ps[:, 1]))[ls]/fac
clee = np.concatenate(([0,0], input_ps[:, 2]))[ls]/fac
clbb = np.concatenate(([0,0], input_ps[:, 3]))[ls]/fac
clte = np.concatenate(([0,0], input_ps[:, 4]))[ls]/fac

isim = args.sim_num

cmb_seed = (isim, 0, 0, 0)
# generate and write CMB_alm.fits
alm = hp.synalm((cltt, clee, clbb, clte), lmax=args.lmax_write, new=True)

filename = cmb_dir + f"/CMB_fullsky_alm_{isim:03d}.fits"
hp.write_alm(filename, np.complex64(alm), overwrite=True)

del alm 





