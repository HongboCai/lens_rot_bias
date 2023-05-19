#!/bin/bash
cd /global/cscratch1/sd/hongbo/lens_rot_bias/src/

python Alpha_sim.py --sim_num=0

for i in {10..19}; do
    python Alpha_sim.py --sim_num=$i
done  
