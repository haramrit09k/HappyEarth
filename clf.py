from PIL import Image
import numpy as np
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def predict2(image_path):

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    classes=['beer-bottle','book', 'can', 'cardboard', 'egg', 'flower', 'food-peels', 'fruit', 'jute', 'leaf', 'meat', 'newspaper', 'paper-plate', 'pizza-box', 'plant', 'plastic-bag', 'plastic-bottle', 'spoilt-food', 'steel-container', 'thermocol']

    model = models.densenet121(pretrained=True)
    # print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, len(classes)),
                                    nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    model.to(device)

    # Load pre-saved model for testing
    model.load_state_dict(torch.load('model_final.pt', map_location=torch.device('cpu')))

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(test_transforms(img), 0)

    model.eval()
    out = model(batch_t)

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:1]]



# def predict(image_path):
#     resnet = models.resnet101(pretrained=True)

#     #https://pytorch.org/docs/stable/torchvision/models.html
#     transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
#     )])

#     img = Image.open(image_path)
#     batch_t = torch.unsqueeze(transform(img), 0)

#     resnet.eval()
#     out = resnet(batch_t)

#     with open('imagenet_classes.txt') as f:
#         classes = [line.strip() for line in f.readlines()]

#     prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
#     _, indices = torch.sort(out, descending=True)
#     return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

