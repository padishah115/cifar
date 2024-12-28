import torch

a = torch.tensor([2.0, 3.0])

_, predicted = torch.max(a, dim=0)

print(_, predicted)