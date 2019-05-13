import torch
from torch.utils.model_zoo import load_url
from torchvision import models

model = './models/vgg16-00b39a1b.pth'
sd = torch.load(model)
# print(sd)
sd['classifier.0.weight'] = sd['classifier.1.weight']
sd['classifier.0.bias'] = sd['classifier.1.bias']
del sd['classifier.1.weight']
del sd['classifier.1.bias']

sd['classifier.3.weight'] = sd['classifier.4.weight']
sd['classifier.3.bias'] = sd['classifier.4.bias']
del sd['classifier.4.weight']
del sd['classifier.4.bias']

torch.save(sd, "vgg16.pth")