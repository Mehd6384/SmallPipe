import time 


class Timer(): 

	def __init__(self): 

		self.cumul = 0.
		self.s = time.time() 
		self.e = 0. 

	def get_time(self): 

		self.e = time.time()
		delta = self.e - self.s 
		self.cumul += delta

		self.s = time.time() 
		return delta, self.cumul 

	def reset(self): 

		self.cumul = 0.
		self.s = time.time() 