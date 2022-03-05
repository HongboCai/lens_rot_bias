import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from pixell import curvedsky

ls, clpp = np.loadtxt('../inputPs/cosmo2017_10K_acc3_lenspotentialCls.dat', usecols=(0,5), unpack=True)
ls = np.concatenate(([0,1],ls))
clpp = np.concatenate(([0,0],clpp))

factor = 0.5*ls**2

phi_alm = curvedsky.rand_alm(clpp, lmax=6000, seed=(0,0,2,0))
kappa_alm = hp.almxfl(phi_alm, factor)

clpp_sim = hp.alm2cl(phi_alm)
clkk_sim = hp.alm2cl(kappa_alm)

hp.write_alm('kappa_fullsky_alm_test_py.fits', kappa_alm, overwrite=True, out_dtype=complex128)
