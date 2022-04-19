import sys, os, time
import numpy as np
import param as p

sim_nums = [i for i in range(10)]
rot_nums = [i for i in range(10)]
nrecon_per_job = 2

recon_sim_job_nums = [i for i in range(int(np.ceil(len(sim_nums*len(p.exps_config)*len(p.moments))/nrecon_per_job)))]
recon_rot_job_nums = [i for i in range(int(np.ceil(len(rot_nums*len(p.exps_config)*len(p.moments))/nrecon_per_job)))]

add_noise = False
qos = 'regular'
time = '08:00:00'
OMP_NUM_THREADS = 4
n_tasks_per_node = int(nrecon_per_job * OMP_NUM_THREADS)

job_path = p.repodir + 'ReconJobs/'
src_path = p.repodir + 'src'

# arguments list for all processes
args_list = []

# recon sim jobs
for experiment, values  in p.exps_config.items():
    for groups, moment in p.moments.items():
        for sim_num in sim_nums:
            args_dict = {}
            args_dict['cmb_map'] = p.repodir + f'Maps/CMBLensed_fullsky_alm_{sim_num:03d}.fits'
            args_dict['experiment'] = experiment
            args_dict['nlev_t'] = values['nlev_t']
            args_dict['nlev_p'] = values['nlev_p']
            args_dict['beam_arcmin'] = values['beam_arcmin']
            args_dict['ellmin' ]= moment['ellmin']
            args_dict['ellmax' ]= moment['ellmax']
            args_dict['delta_L'] = moment['delta_L']
            args_list.append(args_dict)

recon_param_chuncks = np.array_split(args_list, len(recon_sim_job_nums))

# distribute processes to different jobs            
for i, recon_job in enumerate(recon_sim_job_nums):
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
    f.write('cd %s\n\n' %src_path)
    
    # write processes in one job
    if add_noise == True:
        for recon_param in recon_param_chuncks[i]:
            f.write('OMP_NUM_THREADS=%s python recon.py --cmb_map \'%s\' --experiment \'%s\' --nlev_t %s --beam_arcmin %s --ellmin %s --ellmax %s --delta_L %s --add_noise & sleep 1\n' %(OMP_NUM_THREADS, recon_param['cmb_map'], recon_param['experiment'], recon_param['nlev_t'], recon_param['beam_arcmin'], recon_param['ellmin'], recon_param['ellmax'], recon_param['delta_L']))
    else:
        for recon_param in recon_param_chuncks[i]:
            f.write('OMP_NUM_THREADS=%s python recon.py --cmb_map \'%s\' --experiment \'%s\' --nlev_t %s --beam_arcmin %s --ellmin %s --ellmax %s --delta_L %s & sleep 1\n' %(OMP_NUM_THREADS, recon_param['cmb_map'], recon_param['experiment'], recon_param['nlev_t'], recon_param['beam_arcmin'], recon_param['ellmin'], recon_param['ellmax'], recon_param['delta_L']))
    
        
    f.write('wait\n')
    f.close()

# arguments list for all processes
args_list = []

# recon rot jobs
for experiment, values  in p.exps_config.items():
    for groups, moment in p.moments.items():
        for rot_num in rot_nums:
            args_dict = {}
            args_dict['cmb_map'] = p.repodir + f'Maps/CMBLensedRot_fullsky_alm_{rot_num:03d}.fits'
            args_dict['experiment'] = experiment
            args_dict['nlev_t'] = values['nlev_t']
            args_dict['nlev_p'] = values['nlev_p']
            args_dict['beam_arcmin'] = values['beam_arcmin']
            args_dict['ellmin' ]= moment['ellmin']
            args_dict['ellmax' ]= moment['ellmax']
            args_dict['delta_L'] = moment['delta_L']
            args_list.append(args_dict)

recon_param_chuncks = np.array_split(args_list, len(recon_rot_job_nums))

# distribute processes to different jobs            
for i, recon_job in enumerate(recon_rot_job_nums):
    f = open('%s/reconRotJob.sh.%d' %(job_path, recon_job), 'w')

    f.write('#!/bin/bash\n')
    f.write('#SBATCH --qos %s\n' %qos)
    f.write('#SBATCH --constraint=haswell\n')
    f.write('#SBATCH -N 1\n')
    f.write('#SBATCH -t %s\n' %time)
    f.write('#SBATCH --ntasks-per-node=%s\n' %n_tasks_per_node)
    f.write('#SBATCH --license=SCRATCH\n')
    f.write('#SBATCH --output=%sslurmjob.log.%d\n' %(job_path, recon_job))
    f.write('\n')
    f.write('cd %s\n\n' %src_path)
    
    # write processes in one job
    if add_noise == True:
        for recon_param in recon_param_chuncks[i]:
            f.write('OMP_NUM_THREADS=%s python recon.py --cmb_map \'%s\' --experiment \'%s\' --nlev_t %s --beam_arcmin %s --ellmin %s --ellmax %s --delta_L %s --add_noise & sleep 1\n' %(OMP_NUM_THREADS, recon_param['cmb_map'], recon_param['experiment'], recon_param['nlev_t'], recon_param['beam_arcmin'], recon_param['ellmin'], recon_param['ellmax'], recon_param['delta_L']))
    else:
        for recon_param in recon_param_chuncks[i]:
            f.write('OMP_NUM_THREADS=%s python recon.py --cmb_map \'%s\' --experiment \'%s\' --nlev_t %s --beam_arcmin %s --ellmin %s --ellmax %s --delta_L %s & sleep 1\n' %(OMP_NUM_THREADS, recon_param['cmb_map'], recon_param['experiment'], recon_param['nlev_t'], recon_param['beam_arcmin'], recon_param['ellmin'], recon_param['ellmax'], recon_param['delta_L']))
        
    f.write('wait\n')
    f.close()    

            
