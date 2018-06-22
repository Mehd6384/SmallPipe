import torch
import numpy as np 
import pickle 
import matplotlib.pyplot as plt 

def read_data(path, name): 

	return pickle.load(open(path + name, 'rb')) 

class Loader(): 

	def __init__(self, path = '../data/', max_points = 1500, nb_joints = 3, code_size = 8,output_size = 3): 

		self.path = path 
		self.max_points = max_points
		self.code_size = code_size
		self.output_size = output_size
		self.perception_size = nb_joints
		self.counter = 0 

	def sample(self, batch_size = 32): 

		data_p = np.zeros((batch_size, self.perception_size))
		data_t = np.zeros((batch_size, 2))

		label_p = np.zeros((batch_size, self.perception_size)) 
		label_q = np.zeros((batch_size, self.output_size))

		ind = np.random.randint(0,self.max_points, (batch_size))

		for i in range(batch_size): 



			data_p[i,:] = read_data(self.path, 's{}'.format(ind[i]))[:3] # joints angles
			data_t[i,:] = read_data(self.path, 's{}'.format(ind[i]))[-2:] # Target position from DataCreation

			is_over = False if pickle.load(open(self.path + 'eos{}'.format(ind[i]), 'rb')) == 1 else True

			if is_over: # if EndOfSequence next state is same state and joints to 0 
				label_q[i,:] = pickle.load(open(self.path + 'o{}'.format(ind[i]) ,'rb'))*0.
				label_p[i,:] = pickle.load(open(self.path + 's{}'.format(ind[i]) ,'rb'))[:3]
			else: 
				label_q[i,:] = pickle.load(open(self.path + 'o{}'.format(ind[i]) ,'rb'))
				label_p[i,:] = pickle.load(open(self.path + 's{}'.format(ind[i]+1) ,'rb'))[:3]


		return torch.tensor(data_p).float(), torch.tensor(data_t).float(), torch.tensor(label_p).float(), torch.tensor(label_q).float()



class TrajectoryLoader(): 

	def __init__(self, path = '../data/', max_points = 100, nb_joints = 3,  code_size = 4 ): 

		self.path = path 
		self.max_points = max_points
		self.code_size = code_size
		self.output_size = nb_joints
		self.current_point = 0

	def sample(self): 

		ind = self.current_point
		# print('Starting at {}'.format(self.current_point))
		over = False 
		count = 0 
		while not over: 

			over = True if pickle.load(open(self.path + 'eos{}'.format(ind+count), 'rb')) == 0 else False
			count += 1 
		# print('Count is :{}'.format(count))
		p, p_next, q_true = np.zeros((count, self.output_size)), np.zeros((count, self.output_size)), np.zeros((count, self.output_size))
		targ = np.zeros((count, 2))
		for iLoop in range(count): 

			p[iLoop,:]  = read_data(self.path,'s{}'.format(self.current_point + iLoop))[:3]
			targ[iLoop,:] = read_data(self.path,'s{}'.format(self.current_point + iLoop))[:3][-2:]
			p_next[iLoop,:] = read_data(self.path,'s{}'.format(self.current_point + iLoop))[:3]
			q_true[iLoop,:] = read_data(self.path,'o{}'.format(self.current_point + iLoop))[:3]


		self.current_point += count

		return torch.tensor(p).float(), torch.tensor(targ).float(), torch.tensor(p_next).float(), torch.tensor(q_true).float()