import sys, os, time
import numpy as np
import param as p

# jobs of generating all rot lensed CMB maps
rot_nums = [i for i in range(100,110)]
nrot_per_job = 4

rot_job_nums = [i for i in range(int(np.ceil(len(rot_nums)/nrot_per_job)))]
rot_num_chuncks = np.array_split(rot_nums, len(rot_job_nums))

qos = 'regular'
time = '00:30:00'
OMP_NUM_THREADS = 4
ntasks_per_node = int(nrot_per_job * OMP_NUM_THREADS)
job_path = p.repodir + 'RotJobs/'
src_path = p.repodir + 'src/'


# generate rot maps
for i, rot_job in enumerate(rot_job_nums):
    f = open(job_path + 'rot_job_%s.sh' %rot_job, 'w')
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --qos %s\n' %qos)
    f.write('#SBATCH --constraint=haswell\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH -t %s\n' %time)
    f.write('#SBATCH --ntasks-per-node=%s\n' %ntasks_per_node)
    f.write('#SBATCH --license=SCRATCH\n')
    f.write('#SBATCH --output=%sslurmjob.log.%d\n' %(job_path, rot_job))
    f.write('\n')
    f.write('cd %s\n\n' %src_path)
    for rot_num in rot_num_chuncks[i]:
        print('%s: %s'%(rot_job,rot_num))    
        f.write('OMP_NUM_THREADS=%s python CMBLensedRot_sim.py --sim_num=%s & sleep 1\n' %(OMP_NUM_THREADS, rot_num))

    f.write('wait\n')
    f.close()

