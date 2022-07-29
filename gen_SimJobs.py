import sys, os, time
import numpy as np
import param as p

# jobs of generating all sim lensed CMB maps
sim_nums = [i for i in range(100, 110)]
nsim_per_job = 3

sim_job_nums = [i for i in range(int(np.ceil(len(sim_nums)/nsim_per_job)))]
sim_num_chuncks = np.array_split(sim_nums, len(sim_job_nums))

qos = 'regular'
time = '00:30:00'
OMP_NUM_THREADS = 4
ntasks_per_node = int(nsim_per_job * OMP_NUM_THREADS)
job_path = p.repodir + 'SimJobs/'
src_path = p.repodir + 'src/'


# generate sim maps
for i, sim_job in enumerate(sim_job_nums):
    f = open(job_path + 'sim_job_%s.sh' %sim_job, 'w')
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --qos %s\n' %qos)
    f.write('#SBATCH --constraint=haswell\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH -t %s\n' %time)
    f.write('#SBATCH --ntasks-per-node=%s\n' %ntasks_per_node)
    f.write('#SBATCH --license=SCRATCH\n')
    f.write('#SBATCH --output=%sslurmjob.log.%d\n' %(job_path, sim_job))
    f.write('\n')
    f.write('cd %s\n\n' %src_path)
    for sim_num in sim_num_chuncks[i]:
        print('%s: %s'%(sim_job,sim_num))    
        f.write('OMP_NUM_THREADS=%s python CMBLensed_sim.py --sim_num=%s & sleep 1\n' %(OMP_NUM_THREADS, sim_num))

    f.write('wait\n')
    f.close()

