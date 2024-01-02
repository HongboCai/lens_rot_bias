alpha_sim:
	bash src/alpha_sim.sh

make_sim: #alpha_sim
	bash src/sim.sh

rot_sim: #alpha_sim
	bash src/rot_sim.sh
