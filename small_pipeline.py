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
from CommonLoader import Loader
from utils import Timer 

from argparse import ArgumentParser 

parser = ArgumentParser()

parser.add_argument('--data_path', help = 'Path to data', default = "data/")
parser.add_argument('--model_path', help = 'Path to models', default = "models/")
parser.add_argument('--names', help = 'Models name', default = "smalls")
parser.add_argument('--code_size', help = 'Encoding dimension', type = int, default = 4)
parser.add_argument('--nb_im', help = 'Number of datapoints', type = int, default = 12000)
parser.add_argument('--nb_joints', help = 'Number of joints', type = int, default = 3)
parser.add_argument('--validation', help = 'Ask for validation', type = int, default = 1)
parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 32)
parser.add_argument('--lr', help = 'LR ', type = float, default = 3e-4)
parser.add_argument('--mini_epochs', help = 'Batch per epochs', type = int, default = 256)
parser.add_argument('--epochs', help = 'Epochs', type = int, default = 1000)

data_path = parser.parse_args().data_path
model_path = parser.parse_args().model_path
code_size = parser.parse_args().code_size
nb_im = parser.parse_args().nb_im
nb_joints = parser.parse_args().nb_joints
names = parser.parse_args().nb_joints
batch_size = parser.parse_args().batch_size
lr = parser.parse_args().lr
mini_epochs= parser.parse_args().mini_epochs 
epochs= parser.parse_args().epochs 
validation = True if parser.parse_args().validation == 1 else False

checkup = 'Data: {}\nModels: {}\nNames: {}\nCode size: {}\nNb joints: {}\nNb images: {}\nBatch size: {}\nBatches per epoch: {}\nEpochs: {}\nLR: {}'.format(
	data_path,
	model_path,
	names,
	code_size,
	nb_joints,
	nb_im, 
	batch_size, 
	mini_epochs, 
	epochs,
	lr)

print('\n\n\n \t *** Parameters ***\n\n', checkup)
if validation: 
	proceed = input('Proceed ? \t y/n   ')
else: 
	proceed = 'y'

if proceed == 'y': 


	loader = Loader(path = data_path, max_points = nb_im, code_size = code_size, output_size = nb_joints)

	e = EModel(code_size = code_size, in_size = nb_joints)
	t = TaskModel(code_size = code_size)
	c = ControlModel(code_size = code_size, output_size = nb_joints)

	parameters = list(e.parameters()) + list(t.parameters()) + list(c.parameters())
	adam = optim.Adam(parameters, lr)

	def run(m_e): 

		mean_loss = 0. 
		for _ in range(m_e): 

			percep, targ, next_percep, real_q = loader.sample(batch_size)

			e_out = e(percep)
			task_pred = t(e_out, targ)
			control_pred = c(torch.cat([e_out, task_pred],1))

			# input(control_pred.shape)

			task_loss = F.mse_loss(task_pred, e(next_percep))
			control_loss = F.mse_loss(control_pred, real_q)
			global_loss = 0.5*(task_loss + control_loss)

			adam.zero_grad()
			global_loss.backward()
			adam.step()

			mean_loss += global_loss.item()

		return mean_loss/m_e*1.


	timer = Timer()
	for epoch in range(1,epochs+1): 

		l = run(mini_epochs)
		d, cum = timer.get_time()
		print('\t\t *** Epoch {} *** \nGlobal Loss {:.6f}\nLoop time: {:.1f}s - Cum. Time: {}m{:.1f}s'.format(epoch, l, d, int(cum/60), cum%60))


		if epoch%20 == 0: 

			e.save(model_path + 'e_{}'.format(names))
			t.save(model_path + 't_{}'.format(names))
			c.save(model_path + 'c_{}'.format(names))



