import numpy as np 
import matplotlib.pyplot as plt 
import argparse
import pygame as pg 
import pickle

from World import World 
import time 


parser = argparse.ArgumentParser()
parser.add_argument("--frames", help="Number of frames", type = int, default = 100)
parser.add_argument("--joints", help="Number of joints", type = int, default = 3)
parser.add_argument("--length", help="Joints length", type = float, default = 0.2)
parser.add_argument("--obs", help="Number of obstacles", type = int, default = 3)
parser.add_argument("--path", help="Save folder path", default = '../Data/')
parser.add_argument("--frameskip", help="Frame skipping", type = int, default = 3)
parser.add_argument("--dim", help = "Image dimension", type= int, default = 64)
parser.add_argument("--save_deltas", help = "Save also control info", type =bool, default = False)
parser.add_argument("--randomize_obs", help = "Randomize obstacles", type = bool, default = False)
parser.add_argument("--ask_validation", help = "Use validation", type = int, default = 1)
parser.add_argument("--target_limits", help ='Target limits xmin xmax ymin ymax', type = float, nargs = '+', default = [0.2,0.8,0.2,0.8])
parser.add_argument("--use_eos", help ='Use end of sequence', type = int, default = 1)
parser.add_argument("--robot_speed", help ='Robot speed', type = float, default = 1.)
parser.add_argument("--max_steps", help ='Max sequence length', type = int, default = 500)

frames = parser.parse_args().frames 
path = parser.parse_args().path 
frame_skip = parser.parse_args().frameskip
dim = [parser.parse_args().dim,parser.parse_args().dim]
nb_obs = parser.parse_args().obs
save_deltas = parser.parse_args().save_deltas
randomize_obs = parser.parse_args().randomize_obs
validation = parser.parse_args().ask_validation
target_limits = parser.parse_args().target_limits
use_eos = True if parser.parse_args().use_eos == 1 else False 
robot_speed = parser.parse_args().robot_speed
max_steps = parser.parse_args().max_steps
counter = 0 

print('\n\n*****Parameters*****\n')
checkup = 'Frames: {} \nPath: {} \nObstacles: {}\nSaving deltas: {}\nRobot speed: {}\nMax steps: {}\nRandomizing obstacles: {}\nTarget limits: \
 {}\nUse EndOfSequence: {}'.format(frames, path, nb_obs,save_deltas,robot_speed, max_steps, randomize_obs, target_limits, use_eos)

if(validation == 1): 
	proceed = input(checkup + '\n\n\nProceed ? y/n --> ')
else: 
	print(checkup)
	proceed = 'y'


def save_data(state_lists, c): 

	pickle.dump(state_lists[0], open(path+'im{}'.format(c), 'wb'))
	pickle.dump(state_lists[1], open(path +'s{}'.format(c), 'wb'))

def save_output(s, c): 
	pickle.dump(s, open(path+'o{}'.format(c), 'wb'))

def save_eos(e, c): 
	pickle.dump(e, open(path+'eos{}'.format(c), 'wb'))

if proceed == 'y':
	env = World(robot_joints = parser.parse_args().joints, 
				joints_length = parser.parse_args().length, 
				robot_speed = robot_speed,
				randomize_robot = False, 
				randomize_target = True, 
				reset_robot = False, 
				target_limits = target_limits, 
				obstacles = nb_obs, 
				randomize_obstacles = randomize_obs, 
				max_steps = max_steps, 
				grid_cell = 20, 
				path_ready = False)

	env.render()
	frame = 0 
	start = time.time() 
	nb_traj = 0
	while frame < frames: 

		if frame%frame_skip == 0: 
			s1,s2,done = env.observe_image_and_state()
			deltas = env.step_astar(only_deltas = True)
			env.render()

			s2 = env.get_state_elements(['joints_angles', 'joints_pos','target_pos'])

			save_data([s1,s2], counter)
			if save_deltas: 

				if use_eos: 
					mask = 0. if done else 1. 
					save_eos(mask, counter)
					deltas *= mask

				save_output(deltas, counter)

			counter += 1 
			if done: 
				print('{} trajectories for {}/{} frames recorded over {}'.format(nb_traj, counter, frame, frames))
				nb_traj += 1
		else: 
			deltas, obs = env.step_astar()
			# done = obs[1]

		frame += 1
		if done: 
			env.reset()
			


	end = time.time() 
	full_time = end - start
	print('Over with {} observations for {} trajectories\nDone in {}m{:.1f}s'.format(counter, nb_traj, int(full_time/60), full_time%60))
else: 
	print('Canceling. Make sure to hit y to launch')



