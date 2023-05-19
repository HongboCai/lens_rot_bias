#!/bin/bash
cd /global/cscratch1/sd/hongbo/lens_rot_bias/src/

for i in {3..19}; do
    python CMB_sim.py --sim_num=$i
done  
