import sys, os
import healpy as hp, numpy as np
import os, os.path as op
import argparse
import pandas as pd
from datetime import datetime

from orphics import maps, stats, cosmology
from pixell import enmap, utils as u
from enlib import bench

import curvedsky
# add the parent dir in the python path
sys.path.append(os.path.dirname(os.getcwd()))
import param as p
import re

defaults = {
    'mapdir': '../Maps/',
    'odir': '../Maps/',
    'lmax_write': p.lmax_write
}

def log(text):
    time = datetime.now().strftime("%H:%M:%S")
    with open(args.logfile, "a") as f:
        f.write(f"{time}\t{text}\n")

# define parser
parser = argparse.ArgumentParser()

parser.add_argument('--num', type=int, help='result num')
parser.add_argument('--cmb1', type=str, help='the input cmb1')
parser.add_argument('--cmb2', type=str, help='the input cmb2')
parser.add_argument('--cmb3', type=str, help='the input cmb3')
parser.add_argument('--cmb4', type=str, help='the input cmb4')
parser.add_argument('--A_cb', type=float, help='A_cb')

parser.add_argument('--experiment', type=str, help='experiment name')
parser.add_argument("--mapdir",       type=str, default=defaults['mapdir'], help="Output directory")
parser.add_argument('--nlev_t', type=float, help='noise level of temperature field, in ukarcmin', default=7)
parser.add_argument('--beam_arcmin', type=float, help='beam_arcmin', default=1.4)
parser.add_argument('--ellmin', type=int, help='ellmin of CMB', default=30)
parser.add_argument('--ellmax', type=int, help='ellmax of CMB', default=3000)
parser.add_argument('--delta_L', type=int, help='delta_L of Kappa', default=150)
parser.add_argument('--logfile', default='log.txt')
parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--nside', default=1024)        

args = parser.parse_args()
if not(op.exists(defaults['odir'])): os.makedirs(defaults['odir'])

# parse parameters into compatible dict
params = {} # params for lensing reconstruction
params['experiment'] = args.experiment
params['nlev_t'] = args.nlev_t
params['nlev_p'] = args.nlev_t*2**0.5
params['beam_arcmin'] = args.beam_arcmin
params['ellmin'] = args.ellmin
params['ellmax'] = args.ellmax
params['Lmin'] = args.ellmin
params['Lmax'] = args.ellmax

cmb_dir = args.mapdir

lmax = defaults['lmax_write']

# inverse variance filtering
ls = np.arange(0, lmax+1)

nl  = (params['nlev_p']*np.pi/180/60)**2/maps.gauss_beam(ls, params['beam_arcmin'])**2
nl  = np.stack([nl/2, nl, nl, nl*0], axis=0)
theory = cosmology.default_theory()
clee, clbb = theory.lCl('EE', ls), theory.lCl('BB', ls)
nlee = (params['nlev_p']*np.pi/180/60)**2/maps.gauss_beam(ls, params['beam_arcmin'])**2
nlbb = nlee
oclee = clee + nlee
oclbb = clbb + nlbb

oclee[0], oclee[1], oclbb[0], oclbb[1] = 1, 1, 1, 1


log("reading alm...")

imax = params['ellmax'] + 1

# normalization
Al = curvedsky.norm_quad.qeb('lens', params['ellmax'],
                             params['ellmin'], params['ellmax'], clee[:imax], oclee[:imax],
                             oclbb[:imax])

# reconstruction for the two realizations with the same lensing realization
with bench.show("read teb_alm1 and teb_alm2"):
    teb_alm1 = hp.read_alm(args.cmb1, hdu=(1,2,3))
    teb_alm2 = hp.read_alm(args.cmb2, hdu=(1,2,3))

Elm1 = hp.almxfl(teb_alm1[1], 1/oclee)
Blm1 = hp.almxfl(teb_alm1[2], 1/oclbb)
Elm1 = curvedsky.utils.lm_healpy2healpix(Elm1, lmax)
Blm1 = curvedsky.utils.lm_healpy2healpix(Blm1, lmax)


Elm2 = hp.almxfl(teb_alm2[1], 1/oclee)
Blm2 = hp.almxfl(teb_alm2[2], 1/oclbb)
Elm2 = curvedsky.utils.lm_healpy2healpix(Elm2, lmax)
Blm2 = curvedsky.utils.lm_healpy2healpix(Blm2, lmax)
    
reckap_alm_a = curvedsky.rec_lens.qeb(params['ellmax'],
                                      params['ellmin'], params['ellmax'], clee[:imax], Elm1[:imax,:imax],
                                      Blm2[:imax,:imax])
reckap_alm_a *= Al[0][:,None]

reckap_alm_b = curvedsky.rec_lens.qeb(params['ellmax'],
                                      params['ellmin'], params['ellmax'], clee[:imax], Elm2[:imax,:imax],
                                      Blm1[:imax,:imax])
reckap_alm_b *= Al[0][:,None]

term1 = curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_a[0])
term2 = curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_a[0], alm2=reckap_alm_b[0])

del teb_alm1, teb_alm2, Elm1, Blm1, Elm2, Blm2, reckap_alm_a, reckap_alm_b


# reconstruction for the two realizations with the different lensing realizations
with bench.show("read teb_alm2 and teb_alm3"):
    teb_alm3 = hp.read_alm(args.cmb3, hdu=(1,2,3))
    teb_alm4 = hp.read_alm(args.cmb4, hdu=(1,2,3))

Elm3 = hp.almxfl(teb_alm3[1], 1/oclee)
Blm3 = hp.almxfl(teb_alm3[2], 1/oclbb)
Elm3 = curvedsky.utils.lm_healpy2healpix(Elm3, lmax)
Blm3 = curvedsky.utils.lm_healpy2healpix(Blm3, lmax)


Elm4 = hp.almxfl(teb_alm4[1], 1/oclee)
Blm4 = hp.almxfl(teb_alm4[2], 1/oclbb)
Elm4 = curvedsky.utils.lm_healpy2healpix(Elm4, lmax)
Blm4 = curvedsky.utils.lm_healpy2healpix(Blm4, lmax)    

reckap_alm_c = curvedsky.rec_lens.qeb(params['ellmax'],
                                      params['ellmin'], params['ellmax'], clee[:imax], Elm3[:imax,:imax],
                                      Blm4[:imax,:imax])
reckap_alm_c *= Al[0][:,None]

reckap_alm_d = curvedsky.rec_lens.qeb(params['ellmax'],
                                      params['ellmin'], params['ellmax'], clee[:imax], Elm4[:imax,:imax],
                                      Blm3[:imax,:imax])
reckap_alm_d *= Al[0][:,None]

term3 = curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_c[0])
term4 = curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_c[0], alm2=reckap_alm_d[0])

del teb_alm3, teb_alm4, Elm3, Blm3, Elm4, Blm4, reckap_alm_c, reckap_alm_d


# N1aa with two pairs of realizations
N1aa = term1 + term2 - term3 - term4

np.savetxt('../output/N1aa/N1aa_%s_%s_%s_%s_%s'%(args.num, args.A_cb, args.experiment, args.ellmin, args.ellmax)+'.dat', N1aa)




