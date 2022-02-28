#!/bin/bash
#SBATCH --qos regular
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=8.0
#SBATCH --license=SCRATCH
#SBATCH --output=/global/cscratch1/sd/hongbo/lens_rot_bias/jobs/slurmjob.log.2

cd /global/cscratch1/sd/hongbo/lens_rot_bias/src

OMP_NUM_THREADS=4 python CMBLensed_sim.py --sim_num=4
OMP_NUM_THREADS=4 python CMBLensed_sim.py --sim_num=5
wait
