import torch
import torch.nn.functional as F
import math
from torch.distributions import Normal
a = torch.zeros([1,256,5])
b = torch.ones([1,256,5])
c = torch.cat([a,b],dim=2)
print(c.shape)
mu = c[..., :5]
sigma_squared = c[...,5:]
print(mu.shape,sigma_squared.shape)
print(mu)
print(sigma_squared)
print(F.softplus(mu))
print(F.softplus(sigma_squared))
