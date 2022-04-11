import sys, os, time
import numpy as np
import param as p

sim_nums = [i for i in range(10)]
rot_nums = [0,1]
nrecon_per_job = 4


recon_job_nums = [i for i in range(int(np.ceil(len(sim_nums)+len(rot_nums)/nrecon_per_job)))]

recon_num_chuncks = np.array




add_noise = True
obs_filter = True
qos = 'regular'
time = '00:30:00'
OMP_NUM_THREADS = 4
n_tasks_per_node = int(nrecon_per_job * OMP_NUM_THREADS)
job_path = p.repodir + 'reconJobs/'
sim_path = p.repodir + 'src'

# arguments list for all processes
args_list = []

for experiment, values  in p.exps_config.items():
    for groups, moment in p.moments.items():
        for sim_num in sim_nums:
            args_dict = {}
            args_dict['sim_num'] = sim_num
            args_dict['experiment'] = experiment
            args_dict['nlev_t'] = values['nlev_t']
            args_dict['nlev_p'] = values['nlev_p']
            args_dict['beam_arcmin'] = values['beam_arcmin']
            args_dict['ellmin' ]= moment['ellmin']
            args_dict['ellmax' ]= moment['ellmax']
            args_dict['delta_L'] = moment['delta_L']
            args_dict['pure'] = p.pure
            args_list.append(args_dict)        

recon_job_chuncks = np.array_split(args_list, len(recon_jobs))

# distribute processes to different jobs            
for i, recon_job in enumerate(recon_jobs):
    f = open('%s/reconJob.sh.%d' %(job_path, recon_job), 'w')

    f.write('#!/bin/bash\n')
    f.write('#SBATCH --qos %s\n' %qos)
    f.write('#SBATCH --constraint=haswell\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH -t %s\n' %time)
    f.write('#SBATCH --ntasks-per-node=%s\n' %n_tasks_per_node)
    f.write('#SBATCH --license=SCRATCH\n')
    f.write('#SBATCH --output=%sslurmjob.log.%d\n' %(job_path, recon_job))
    f.write('\n')
    f.write('cd %s\n\n' %sim_path)
    
    # write processes in one job
    for recon_param in recon_job_chuncks[i]:
        f.write('OMP_NUM_THREADS=%s python recon.py --sim_num %s --experiment \'%s\' --nlev_t %s --beam_arcmin %s --ellmin %s --ellmax %s --delta_L %s --pure %s --add_noise --obs_filter & sleep 1\n' %(OMP_NUM_THREADS, recon_param['sim_num'], recon_param['experiment'], recon_param['nlev_t'], recon_param['beam_arcmin'], recon_param['ellmin'], recon_param['ellmax'], recon_param['delta_L'], recon_param['pure']))
        
    f.write('wait\n')
    f.close()

            
