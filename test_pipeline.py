import torch
import torch.nn.functional as F 
import torch.optim as optim 

import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')


from e_model import EModel
from t_model import TaskModel
from c_model import ControlModel
from CommonLoader import TrajectoryLoader


from argparse import ArgumentParser 

parser = ArgumentParser()

parser.add_argument('--data_path', help = 'Path to data', default = "data/")
parser.add_argument('--model_path', help = 'Path to models', default = "models/")
parser.add_argument('--names', help = 'Models name', default = "smalls")
parser.add_argument('--code_size', help = 'Encoding dimension', type = int, default = 4)
parser.add_argument('--nb_im', help = 'Number of datapoints', type = int, default = 12000)
parser.add_argument('--nb_joints', help = 'Number of joints', type = int, default = 3)
parser.add_argument('--validation', help = 'Ask for validation', type = int, default = 1)

data_path = parser.parse_args().data_path
model_path = parser.parse_args().model_path
code_size = parser.parse_args().code_size
nb_im = parser.parse_args().nb_im
nb_joints = parser.parse_args().nb_joints
names = parser.parse_args().nb_joints
validation = True if parser.parse_args().validation == 1 else False

checkup = 'Data: {}\nModels: {}\nNames: {}\nCode size: {}\nNb joints: {}\nNb images: {}\n'.format(
	data_path,
	model_path,
	names,
	code_size,
	nb_joints,
	nb_im)
print('\n\n\n \t *** Parameters ***\n\n', checkup)

end_process = False 
f, ax = plt.subplots(1, nb_joints)



loader = TrajectoryLoader(path = data_path, max_points = nb_im, code_size = code_size, nb_joints = nb_joints)

e = EModel(code_size = code_size, in_size = nb_joints)
t = TaskModel(code_size = code_size)
c = ControlModel(code_size = code_size, output_size = nb_joints)

e.load(model_path + 'e_{}'.format(names))
t.load(model_path + 't_{}'.format(names))
c.load(model_path + 'c_{}'.format(names))


while not end_process: 

	current_p, target, next_p, real_q = loader.sample()

	code_p = e(current_p)
	task_pred = t(code_p, target)
	control_pred = c(torch.cat([code_p, task_pred], 1)).detach().numpy().T

	q = real_q.detach().numpy().T
	for i in range(nb_joints): 

		ax[i].plot(q[i,:], label = 'Real')
		ax[i].plot(control_pred[i,:], label = 'Pred')
		ax[i].set_title('Joint {}'.format(i))
		ax[i].legend()

	plt.pause(0.1)

	end_process = input('Press o to quit')
	if end_process == 'o': 
		end_process = True
	for a in ax: 
		a.clear()