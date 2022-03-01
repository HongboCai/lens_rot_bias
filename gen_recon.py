import sys, os, time
import numpy as np
import param as p


sim_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
nrecon_per_job = 4

qos = 'regular'
time = '00:30:00'
OMP_NUM_THREADS = 4
n_tasks_per_node = int(nsim_per_job * OMP_NUM_THREADS)
job_path = p.repodir + 'reconJobs/'
sim_path = p.repodir + 'src'

# arguments list for all processes
args_list = []

for experiment, values  in p.exps_config.items():
    for sim_num in sim_nums:
        for groups, moment in p.moments.items():
            args_dict = {}
            args_dict['sim_num'] = sim_nums
            args_dict['experiment'] = experiment
            args_dict['nlev_t'] = values['nlev_t']
            args_dict['nlev_p'] = values['nlev_p']
            args_dict['beam_arcmin'] = values['beam_arcmin']
            args_dict['ellmin' ]= moment['ellmin']
            args_dict['ellmax' ]= moment['ellmax']
            args_dict['delta_L'] = moment['delta_L']
            args_dict['pure'] = p.pure
            args_list.append(args_dict)        

recon_job_chuncks = np.array_split(args_list, nrecon_per_job)

            
