import torch
a = torch.zeros(4,3,6)
b = torch.zeros(4,5,6)
# c = torch.cat([a,b],dim=2)
d = torch.cat([a,b],dim=1)
print(d.shape)