import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim


class TaskModel(nn.Module): 

	def __init__(self, code_size): 

		nn.Module.__init__(self)

		self.l1 = nn.Linear(code_size, 64)
		self.l2 = nn.Linear(66,66)
		self.l3 = nn.Linear(66,code_size)

	def forward(self, x, y): 

		x = F.elu(self.l1(x))
		x = torch.cat([x,y], dim = 1)
		x = F.elu(self.l2(x))

		return F.tanh(self.l3(x))

	def save(self, path): 

		torch.save(self.state_dict(), path)

	def load(self, path):

		self.load_state_dict(torch.load(path))
		