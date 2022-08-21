"""fullsky reconstruction based on cmblensplus, with mpi acceleration"""

import sys, os, re
import healpy as hp, numpy as np
import os, os.path as op
import argparse
import pandas as pd
import itertools

from orphics import maps, cosmology
# from pixell import enmap, utils as u, mpi
from pixell import utils as u

import curvedsky
# add the parent dir in the python path
sys.path.append(os.path.dirname(os.getcwd()))
import param as p

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def create_shared_array_like(arr, comm, shape=None):
    """Create an MPI shared memory with similar shape as input arr. Allow shape overwrite"""
    if shape is None:  shape = arr.shape
    if comm.Get_rank() == 0: nbytes = np.prod(shape)*arr.itemsize
    else: nbytes = 0
    win = MPI.Win.Allocate_shared(nbytes, arr.itemsize, comm=comm) 
    buf, _ = win.Shared_query(0) 
    new_arr = np.ndarray(buffer=buf, dtype=arr.dtype, shape=shape)
    return new_arr


defaults = {
    'mapdir': '../Maps/',
    'odir': '../Maps/',
    'lmax_write': p.lmax_write
}


# define parser
parser = argparse.ArgumentParser()

parser.add_argument('--cmb_map', type=str, help='the input cmb')
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
parser.add_argument('--rdn0_set', type=int, help='rdn0 set number')

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

imax = params['ellmax'] + 1  # inclusive
cmb_dir = args.mapdir
lmax = defaults['lmax_write']

teb_alm = hp.read_alm(args.cmb_map, hdu=(1,2,3))

# read in kappa map for cross_cl
kap_alm = hp.read_alm(defaults['odir'] + 'kappa_fullsky_alm_' + re.split('alm_|/|.fits', args.cmb_map)[-2] + '.fits')
kap_alm = curvedsky.utils.lm_healpy2healpix(kap_alm, lmax)

# generate noise realization
if args.add_noise:
    ls  = np.arange(lmax+1)
    nl  = (params['nlev_p']*np.pi/180/60)**2/maps.gauss_beam(ls, params['beam_arcmin'])**2
    nl  = np.stack([nl/2, nl, nl, nl*0], axis=0)
    nlm = hp.synalm(nl, new=True, lmax=lmax)
else:
    nlm = 0
    nl  = [0,0,0,0]

# inverse variance filtering
ls = np.arange(0, lmax+1)

theory = cosmology.default_theory()
clee, clbb = theory.lCl('EE', ls), theory.lCl('BB', ls)
nlee = (params['nlev_p']*np.pi/180/60)**2/maps.gauss_beam(ls, params['beam_arcmin'])**2
nlbb = nlee
oclee = clee + nlee
oclbb = clbb + nlbb

oclee[0], oclee[1], oclbb[0], oclbb[1] = 1, 1, 1, 1

if args.add_noise:
    Elm = hp.almxfl(teb_alm[1]+nlm[1], 1/oclee)
    Blm = hp.almxfl(teb_alm[2]+nlm[2], 1/oclbb)

else:
    Elm = hp.almxfl(teb_alm[1], 1/oclee)
    Blm = hp.almxfl(teb_alm[2], 1/oclbb)

# convert alm to healpix order to be compatible with clp
Elm = curvedsky.utils.lm_healpy2healpix(Elm, lmax)[:imax,:imax]
Blm = curvedsky.utils.lm_healpy2healpix(Blm, lmax)[:imax,:imax]

# reconstruction
reckap_alm = curvedsky.rec_lens.qeb(params['ellmax'],
                                    params['ellmin'], params['ellmax'], clee[:imax], Elm,
                                    Blm, gtype='k')

# normalization
Al = curvedsky.norm_quad.qeb('lens', params['ellmax'],
                            params['ellmin'], params['ellmax'], clee[:imax], oclee[:imax],
                            oclbb[:imax], lfac='k')# , clbb[:imax])

reckap_alm *= Al[0][:,None]

reckap_cl = curvedsky.utils.alm2cl(params['ellmax'], reckap_alm[0])
cross_cl = curvedsky.utils.alm2cl(params['ellmax'], kap_alm[:imax,:imax],reckap_alm[0])

data_dict = {}
data_dict['reckap_cl'] = reckap_cl
data_dict['cross_cl'] = cross_cl

# calculate RDN0
if args.rdn0_set == 0:
   sims1 = [f'../Maps/CMBLensed_fullsky_alm_{i:03d}.fits' for i in range(10,35)]
   sims2 = [f'../Maps/CMBLensed_fullsky_alm_{i:03d}.fits' for i in range(60,85)]
if args.rdn0_set == 1:
   sims1 = [f'../Maps/CMBLensed_fullsky_alm_{i:03d}.fits' for i in range(10,35)]
   sims2 = [f'../Maps/CMBLensed_fullsky_alm_{i:03d}.fits' for i in range(85,110)]
if args.rdn0_set == 2:
   sims1 = [f'../Maps/CMBLensed_fullsky_alm_{i:03d}.fits' for i in range(35,60)]
   sims2 = [f'../Maps/CMBLensed_fullsky_alm_{i:03d}.fits' for i in range(60,85)]
if args.rdn0_set == 3:
   sims1 = [f'../Maps/CMBLensed_fullsky_alm_{i:03d}.fits' for i in range(35,60)]
   sims2 = [f'../Maps/CMBLensed_fullsky_alm_{i:03d}.fits' for i in range(85,110)]

# create a shared memory for Elm1_sims and Blm1_sims
Elm1_sims = create_shared_array_like(Elm, comm, shape=(len(sims1),)+Elm.shape)
Blm1_sims = create_shared_array_like(Blm, comm, shape=(len(sims1),)+Blm.shape)
if rank == 0:
    for i in range(len(sims1)):
        sim1 = sims1[i]
        if rank == 0: print(f"Loading: {sim1}")
        teb1_alm = hp.read_alm(sim1, hdu=(1,2,3))
        # map noise realization?
        if args.add_noise:
            nlm = hp.synalm(nl, new=True, lmax=lmax)
            Elm1 = hp.almxfl(teb1_alm[1]+nlm[1], 1/oclee)
            Blm1 = hp.almxfl(teb1_alm[2]+nlm[2], 1/oclbb)
        else:
            Elm1 = hp.almxfl(teb1_alm[1], 1/oclee)
            Blm1 = hp.almxfl(teb1_alm[2], 1/oclbb)
        
        Elm1_sims[i,...] = curvedsky.utils.lm_healpy2healpix(Elm1, lmax)[:imax,:imax]
        Blm1_sims[i,...] = curvedsky.utils.lm_healpy2healpix(Blm1, lmax)[:imax,:imax]
    del teb1_alm, Elm1, Blm1
comm.Barrier()

Elm2_sims = create_shared_array_like(Elm, comm, shape=(len(sims2),)+Elm.shape)
Blm2_sims = create_shared_array_like(Blm, comm, shape=(len(sims2),)+Blm.shape)
if rank == 0:
    for i in range(len(sims2)):
        sim2 = sims2[i]
        if rank == 0: print(f"Loading: {sim2}")
        teb2_alm = hp.read_alm(sim2, hdu=(1,2,3))
        # map noise realization?
        if args.add_noise:
            nlm = hp.synalm(nl, new=True, lmax=lmax)
            Elm2 = hp.almxfl(teb2_alm[1]+nlm[1], 1/oclee)
            Blm2 = hp.almxfl(teb2_alm[2]+nlm[2], 1/oclbb)
        else:
            Elm2 = hp.almxfl(teb2_alm[1], 1/oclee)
            Blm2 = hp.almxfl(teb2_alm[2], 1/oclbb)
        
        Elm2_sims[i,...] = curvedsky.utils.lm_healpy2healpix(Elm2, lmax)[:imax,:imax]
        Blm2_sims[i,...] = curvedsky.utils.lm_healpy2healpix(Blm2, lmax)[:imax,:imax]

    del teb2_alm, Elm2, Blm2
comm.Barrier()

if rank == 0: print('calculating rdn0')
if rank == 0: print('=> calculate part 1')
part1s = []
for i in range(rank, len(Elm1_sims), size):
    print(f"{i:>2d}/{len(Elm1_sims)}")
    reckap_alm_a = curvedsky.rec_lens.qeb(params['ellmax'],
                                          params['ellmin'], params['ellmax'], clee[:imax], Elm,
                                          Blm1_sims[i], gtype='k')
    reckap_alm_a *= Al[0][:,None]
    reckap_alm_b = curvedsky.rec_lens.qeb(params['ellmax'],
                                            params['ellmin'], params['ellmax'], clee[:imax], Elm1_sims[i],
                                            Blm, gtype='k')
    reckap_alm_b *= Al[0][:,None]
    part1 = curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_a[0]) + \
            curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_a[0], reckap_alm_b[0]) + \
            curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_b[0]) + \
            curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_b[0], reckap_alm_a[0])
    part1s.append(part1)
part1s = u.allgatherv(part1s, comm)
del part1, reckap_alm_a, reckap_alm_b

if rank == 0: print('=> calculate part 2')
rdn0 = []
pairs = list(itertools.product(range(len(Elm1_sims)), range(len(Elm2_sims))))
for pair_idx in range(rank, len(pairs), size):
    i, j = pairs[pair_idx]
    print(f"{pair_idx:>3d}/{len(pairs)}: i = {i} j = {j}")
    reckap_alm_c = curvedsky.rec_lens.qeb(params['ellmax'],
                                        params['ellmin'], params['ellmax'], clee[:imax], Elm1_sims[i],
                                        Blm2_sims[j], gtype='k')
    reckap_alm_c *= Al[0][:,None]
    reckap_alm_d = curvedsky.rec_lens.qeb(params['ellmax'],
                                        params['ellmin'], params['ellmax'], clee[:imax], Elm2_sims[j],
                                        Blm1_sims[i], gtype='k')
    reckap_alm_d *= Al[0][:,None]
        
    part2 = curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_c[0]) + \
            curvedsky.utils.alm2cl(params['ellmax'], reckap_alm_d[0], reckap_alm_c[0])

    rdn0.append(part1s[i]-part2)
rdn0 = u.allgatherv(rdn0, comm)
rdn0 = np.mean(np.array(rdn0), axis=0)

data_dict['rdn0'] = rdn0
data_df = pd.DataFrame(data_dict)
ofile = '../output/recon_ps/reckap_cl_'+re.split('Maps|/|.fits', args.cmb_map)[-2]+'_%s_%s_%s'%(args.rdn0_set, args.ellmin, args.ellmax)+'.csv'
if rank == 0: print("Writing:", ofile)
data_df.to_csv(ofile, index=False) 