"""This script takes some input rotation power spectrum and generates gaussian
realisations of of the rotation field. The rotation field will be stored as a
fits file storing the relevant alms.

"""
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
parser.add_argument("-o","--odir",default=param.mapdir, help='alm output path')
parser.add_argument("--psdir",default=param.psdir, help='ps output path')
parser.add_argument("--ps", help="input rotational power spectrum", default=None)
parser.add_argument("--ACB", help="optionally specify a scale invariant spectrum by A_CB, in 1e-5", type=float, default=1e-5)
parser.add_argument("--oname", help='output filename', default='alpha_alm_fullskyfits')
parser.add_argument("--lmax", help='maximum ell to generate', type=int, default=10000)
parser.add_argument("--seed", help='seed for realisations', type=int, default=None)

args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)

# load the power spectrum
# this assumes that there is no prefactor in the power spectrum
if args.ps: ls, claa = np.loadtxt(args.ps, usecols=(0,1), unpack=True)
elif args.ACB:
    ls = np.arange(args.lmax)
    claa = 2*np.pi/(ls+1)/ls * args.ACB   # note we are using 1e-5
    claa[0] = 0
assert claa[0] == 0

# lmax has to be the minimum of specified and the input file
lmax = min(args.lmax, ls[-1])
print(f"Generating sims: lmax={lmax}, seed={args.seed}")
alpha_alms = curvedsky.rand_alm_healpy(claa, lmax=lmax, seed=args.seed)
claa_sim = hp.alm2cl(alpha_alms)

# write out alm
print("Writing alpha_alm")
hp.write_alm(args.odir + 'alpha_alms_fullsky.fits', alpha_alms, overwrite=True)

# plot claa
plt.plot(ls, claa, label='data')
plt.plot(ls, claa_sim, label='sim')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('L')
#plt.ylabel('$C_L^{\alpha\alpha}$')
plt.legend()
ofile = op.join(args.psdir, 'claa.pdf')
print("Writing:", ofile)
plt.savefig(ofile, bbox_inches='tight')
plt.close()




