#!/bin/bash
cd /global/cscratch1/sd/hongbo/lens_rot_bias/src/

for i in {0..9}; do
    python CMBRot_sim.py --sim_num=$i --alpha_num=0
done  

for i in {10..19}; do
    python CMBRot_sim.py --sim_num=$i  --alpha_num=$i
done  
