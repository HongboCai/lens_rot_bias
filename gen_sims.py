import sys, os, time
import numpy as np
import param as p

sim_nums = [2, 3, 4, 5]
sim_jobs = [1, 2]
nsim_per_job = np.ceil(len(sim_nums)/len(sim_jobs))

sim_num_chuncks = np.array_split(sim_nums, nsim_per_job)
qos = 'regular'
time = '00:30:00'
OMP_NUM_THREADS = 4
n_tasks_per_node = nsim_per_job * OMP_NUM_THREADS
job_path = p.repodir + 'jobs/'
sim_path = p.repodir + 'src'
    
for i, sim_job in enumerate(sim_jobs):
    f = open(job_path + 'sim_job_%s.sh' %sim_job, 'w')
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --qos %s\n' %qos)
    f.write('#SBATCH --constraint=haswell\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH -t %s\n' %time)
    f.write('#SBATCH --ntasks-per-node=%s\n' %n_tasks_per_node)
    f.write('#SBATCH --license=SCRATCH\n')
    f.write('#SBATCH --output=%sslurmjob.log.%d\n' %(job_path, sim_job))
    f.write('\n')
    f.write('cd %s\n\n' %sim_path)
    for sim_num in sim_num_chuncks[i]:
        print('%s: %s'%(sim_job,sim_num))    
        f.write('OMP_NUM_THREADS=%s python CMBLensed_sim.py --sim_num=%s\n' %(OMP_NUM_THREADS, sim_num))
    f.write('wait\n')
    f.close()
