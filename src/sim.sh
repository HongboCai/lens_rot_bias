#!/bin/bash
cd /global/cscratch1/sd/hongbo/lens_rot_bias/src/

for i in {0..19}; do
    python CMB_sim.py --sim_num=$i
done  
