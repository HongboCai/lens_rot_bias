"""This script generate CMB simulations with rotational field included"""

import os, sys
import os.path as op
import numpy as np
import healpy as hp
from pixell import enmap, utils as u, curvedsky
import argparse
from matplotlib import pyplot as plt

# add the parent dir in the python path
sys.path.append(os.path.dirname(os.getcwd()))
import param

parser = argparse.ArgumentParser()
parser.add_argument("-o","--odir",default=param.outmapdir, help='alm output path')
parser.add_argument("--infile", ,default=param.fullskyLensedCMB_alm, help="cmb alm fits file")
parser.add_argument("--alpha", help="alpha alm fits file")
parser.add_argument("--pix-size", help='pixel size in arcmin', type=float, default=0.5)
parser.add_argument("--oname", default=None)
parser.add_argument("--lmax", type=int, default=6000)
parser.add_argument("--save-original", action='store_true')
args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)

# define a geometry to work with
oshape, owcs = enmap.fullsky_geometry(args.pix_size*u.arcmin)

# load alpha alm and project to map space
print("Loading alpha alm")
alpha_alm = hp.read_alm(args.alpha)
print("Generating alpha map")
alpha_map = enmap.zeros((1,)+oshape, owcs)
curvedsky.alm2map(alpha_alm, alpha_map, spin=0)
del alpha_alm

# actually load cmb map alm
print("Loading cmb alm")
ialm = hp.read_alm(args.infile, hdu=(1,2,3))
# convert to map space
print("Generating cmb map")
imap = enmap.zeros((3,)+oshape, owcs)
curvedsky.alm2map(ialm, imap)
del ialm

# rotate polarization
print("Rotating polarization by alpha")
omap = enmap.rotate_pol(imap, alpha_map)

# convert to alm
print("Converting map to alm")
oalm = curvedsky.map2alm(omap, lmax=args.lmax)
ialm = curvedsky.map2alm(imap, lmax=args.lmax)
del imap

if args.oname is None: oname = bname[:-5]+'_rot.fits'
else: oname = args.oname
ofile = op.join(args.odir, oname)
print("Writing:", ofile)
hp.write_alm(ofile, oalm, overwrite=True)
# also save the original map2alm
if args.save_original:
    ofile = op.join(args.odir, 'fullskyLensedCMB_alm.fits')
    print("Writing:", ofile)
    hp.write_alm(ofile, ialm, overwrite=True)
