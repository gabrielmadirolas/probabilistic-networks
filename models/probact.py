'''
This is (heavily) modified by Gabriel Madirolas, from the code available here:
https://github.com/kumar-shridhar/ProbAct-Probabilistic-Activation-Function
Which is the implementation of this paper:
Shridhar, K., Lee, J., Hayashi, H., Mehta, P., Iwana, B. K., Kang, S., ... & Dengel, A. (2019). 
Probact: A probabilistic activation function for deep neural networks. arXiv preprint arXiv:1905.10761.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
# import torch.jit as jit


#device = torch.device("cuda:0")

class TrainableSigma(nn.Module):


	def __init__(self, num_parameters=1, init_sigma=0): # Gab: init_sigma was called init
		self.num_parameters = num_parameters
		super(TrainableSigma, self).__init__()
		self.sigma = Parameter(torch.Tensor(num_parameters).fill_(init_sigma)) # Gab: sigma was called 'weight'

	def forward(self, x): # Gab: x was originally named 'input'

		mu = x

		if mu.is_cuda:
			eps = torch.cuda.FloatTensor(mu.size()).normal_(mean = 0, std = 1)
		else:
			eps = torch.FloatTensor(mu.size()).normal_(mean = 0, std = 1)

		return F.relu(mu) + self.sigma * eps

# Gab: new class to train a global mu as well
class TrainableMuSigma(nn.Module):


	def __init__(self, init_mu=0, init_sigma=0): # Gab: init_sigma was called init. Removed num_parameters=1
		#self.num_parameters = num_parameters
		super(TrainableMuSigma, self).__init__()
		self.mu = Parameter(torch.tensor(init_mu))
		self.sigma = Parameter(torch.tensor(init_sigma)) # Gab: sigma was called 'weight'
		'''
		self.mu = Parameter(torch.Tensor(num_parameters).fill_(init_mu))
		self.sigma = Parameter(torch.Tensor(num_parameters).fill_(init_sigma)) # Gab: sigma was called 'weight'
		'''
	def forward(self, x): # Gab: x was originally named 'input'

		if x.is_cuda:
			eps = torch.cuda.FloatTensor(x.size()).normal_(mean = 0, std = 1)
		else:
			eps = torch.FloatTensor(x.size()).normal_(mean = 0, std = 1)

		return  self.mu + F.relu(x) + self.sigma * eps
	
# Gab: New class, to have an element-wise trainable sigma
class EWTrainableMuSigma(nn.Module):

	def __init__(self, num_parameters, prob_params):
		#self.num_parameters = num_parameters
		#self.prob_params = prob_params
		super(EWTrainableMuSigma, self).__init__()
		#self.mu = Parameter(torch.mul(torch.ones(tuple(num_parameters)),init_sigma))
		if prob_params["std_mu"]:
			self.mu = Parameter(torch.empty(tuple(num_parameters)).normal_(mean=prob_params["mean_mu"], std=prob_params["std_mu"]))
		else:
			self.mu = prob_params["mean_mu"]	
		self.sigma = Parameter(torch.empty(tuple(num_parameters)).normal_(mean=prob_params["mean_sigma"], std=prob_params["std_sigma"]))
		#self.sigma = Parameter(torch.Tensor(num_parameters).fill_(init_sigma))
		#self.alpha = Parameter(torch.tensor(kwargs["alpha"]))
		#self.beta = Parameter(torch.tensor(kwargs["beta"]))
		self.alpha = prob_params["alpha"]
		self.beta = prob_params["beta"]
		print("sigma requires grad",self.sigma.requires_grad)

	def forward(self, x: Tensor) -> Tensor:
		'''
		mu = x

		if mu.is_cuda:
			eps = torch.cuda.FloatTensor(mu.size()).normal_(mean = 0, std = 1)
		else:
			eps = torch.FloatTensor(mu.size()).normal_(mean = 0, std = 1)
		'''
		if self.sigma.is_cuda:
			#eps = torch.cuda.FloatTensor(x.size()).normal_(mean = 0, std = 1)
			eps = torch.normal(mean = 0.0, std = 1.0, size = (x.size(0),x.size(1))).to('cuda:0')
		else:
			#eps = torch.FloatTensor(x.size()).normal_(mean = 0, std = 1)
			eps = torch.normal(mean = 0.0, std = 1.0, size = (x.size(0),x.size(1)))

		# this is the most general trainable activation function, which includes a trainable sigma if
		# its paramaters alpha and beta are passed to the constructor
		if self.alpha and self.beta:
			return self.mu + self.alpha*torch.sigmoid(self.beta*self.sigma) * eps
		else:
			return self.mu + self.sigma * eps

		#return x
		#return F.relu(x)
		#return self.mu + F.relu(x)
		#return F.relu(mu) 
		#return F.relu(mu) + self.sigma* eps
		#return self.mu + x + self.sigma * eps
		#return F.relu(x)  + self.sigma * eps
		#return F.relu(x) + torch.abs(self.sigma) * eps
		#return self.mu + F.relu(x) + torch.abs(self.sigma) * eps
		#return self.mu  + F.relu(x) + self.alpha*torch.sigmoid(self.beta*self.sigma) * eps
		#return F.relu(x) + self.alpha*torch.sigmoid(self.beta*self.sigma) * eps
		#return self.mu + x + self.alpha*torch.sigmoid(self.beta*self.sigma) * eps
		#return self.mu + x + torch.abs(self.sigma) * eps