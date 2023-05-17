#!/bin/bash
#SBATCH --qos debug
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=16
#SBATCH --license=SCRATCH
#SBATCH --output=/global/cscratch1/sd/hongbo/lens_rot_bias/RotJobs/slurmjob.log.0

cd /global/cscratch1/sd/hongbo/lens_rot_bias/src/

OMP_NUM_THREADS=4 python CMBLensedRot_sim.py --sim_num=0 & sleep 1
OMP_NUM_THREADS=4 python CMBLensedRot_sim.py --sim_num=1 & sleep 1
OMP_NUM_THREADS=4 python CMBLensedRot_sim.py --sim_num=2 & sleep 1
OMP_NUM_THREADS=4 python CMBLensedRot_sim.py --sim_num=3 & sleep 1
wait
