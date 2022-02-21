"""get lensing(kappa and phi) map from cls"""

import os, sys
import os.path as op
import numpy as np
import healpy as hp
import argparse
from matplotlib import pyplot as plt
from pixell import curvedsky

import camb
from camb import model, initialpower

# add the parent dir in the python path
sys.path.append(os.path.dirname(os.getcwd()))
import param

parser = argparse.ArgumentParser()
# alm output path
parser.add_argument("-o","--odir",default=param.mapdir)
# ps output path
parser.add_argument("--psdir",default=param.psdir)

# input phi power spectrum
parser.add_argument("--ps", default=param.lenspotental_cl)

args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)

# get lensing power spectrum
ls, clpp = np.loadtxt(args.ps, usecols=(0,5), unpack=True)

# generate phi and kappa alm
phi_alms = curvedsky.rand_alm(clpp, seed=(0,0,2,0))
clpp_sim = hp.alm2cl(phi_alms)

factor = 0.5 * ls**2
clkk = clpp * factor**2
kappa_alms = hp.almxfl(phi_alms, factor)
clkk_sim = hp.alm2cl(kappa_alms)

hp.write_alm(args.odir + 'phi_alms_fullsky.fits', phi_alms, overwrite=True)
hp.write_alm(args.odir + 'kappa_alms_fullsky.fits', kappa_alms, overwrite=True)


# plot clpp
plt.plot(ls, clpp, label='data')
plt.plot(ls, clpp_sim, label='sim')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
plt.ylabel('$C_L^{\phi\phi}$')
plt.legend()
ofile = op.join(args.psdir, 'clpp.pdf')
print("Writing:", ofile)
plt.savefig(ofile, bbox_inches='tight')
plt.close()

# plot kappa field
plt.plot(ls, clkk, label='data')
plt.plot(ls, clkk_sim, label='sim')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
plt.ylabel('$C_L^{\kappa\kappa}$')
plt.legend()
ofile = op.join(args.psdir, 'clkk.pdf')
print("Writing:", ofile)
plt.savefig(ofile, bbox_inches='tight')

