import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

from .lenet import LeNet5


def mnist_classifier():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = LeNet5().eval()
    net.load_state_dict(torch.load('weights/lenet_epoch=12_test_acc=0.991.pth'))
    net = net.to(device)
    return net