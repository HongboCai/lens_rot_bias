#!/bin/bash
#SBATCH --qos regular
#SBATCH --constraint=haswell
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --license=SCRATCH
#SBATCH --output=/global/cscratch1/sd/hongbo/lens_rot_bias/simJobs/slurmjob.log.11

cd /global/cscratch1/sd/hongbo/lens_rot_bias/src

OMP_NUM_THREADS=4 python CMBLensed_sim.py --sim_num=22 & sleep 1
OMP_NUM_THREADS=4 python CMBLensed_sim.py --sim_num=23 & sleep 1
wait
OMP_NUM_THREADS=4 python CMBLensedRot_sim.py --sim_num=22 & sleep 1
OMP_NUM_THREADS=4 python CMBLensedRot_sim.py --sim_num=23 & sleep 1
wait
