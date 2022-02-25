"""This is a test script that generates some simple scale invariant
rotation power spectrum for testing"""

import argparse
import os, os.path as op
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--odir', default='../input/ps')
parser.add_argument('--ACB', help='amplitude (A_CB) of the signal in unit of 1', type=float, default=1)
parser.add_argument('--lmax', type=int, default=10000)
parser.add_argument('--oname', default='claa.txt')
args = parser.parse_args()
if not op.exists(args.odir): os.makedirs(args.odir)

ell = np.arange(0, args.lmax+1)
ps_alpha = args.ACB*2*np.pi/(ell*(ell+1))
ps_alpha[0] = 0
ofile = op.join(args.odir, args.oname)
print("Writing:", ofile)
np.savetxt(ofile, np.array([ell, ps_alpha]).T, header='ell claa')
