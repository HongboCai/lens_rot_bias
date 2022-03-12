import math

# repository path
repodir = '/global/cscratch1/sd/hongbo/lens_rot_bias/'

# input path and input power spectra files
indir = repodir + 'input/'
lenspotental_cl = indir + 'ps/cosmo2017_10K_acc3_lenspotentialCls.dat'

# output path and output alms files
outdir = repodir + 'output/'

# geometry parameters
px_arcmin = 1.
lmax = 6000
lmax_write = 6000


# loop parameters
exps_config = {
    'CMB_S3': {
        'nlev_t': 7,
        'nlev_p': 7*2**0.5,
        'beam_arcmin': 1.4
    }
}

# exps_config = {
#     'Planck_SMICA': {
#         'nlev_t': 45,
#         'nlev_p':45*2**0.5,
#         'beam_arcmin': 5
#     },
#     'CMB_S3': {
#         'nlev_t': 7,
#         'nlev_p': 7*2**0.5,
#         'beam_arcmin': 1.4
#     },
#     'CMB_S4': {
#         'nlev_t': 1,
#         'nlev_p':2**0.5,
#         'beam_arcmin': 3
#     }
# }

# moments = {'moments1':{'ellmin':30, 'ellmax':3000, 'delta_L':150},'moments2':{'ellmin':30, 'ellmax':4000, 'delta_L':200}}

moments = {'moments1':{'ellmin':30, 'ellmax':3000, 'delta_L':150}}

# 'pure', 'no', default='standard'
pure = 'no'
add_noise = True

# sbatch parameters for each job file
qos = 'regular' # 'regular' or 'debug'
nodes = 1 # N
time = '03:00:00' # t
n_tasks_per_node = 60 # how many threads(one task for one threads)
OMP_NUM_THREADS = 12 # how many threads for one process

# processes and jobs
# job_path = repo_path + 'jobs2/'
# npros = len(A)*len(exps_config)*len(moments) # number of total process
# npros_per_job = 5
# njobs = math.ceil(npros/npros_per_job)


