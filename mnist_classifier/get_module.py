import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

from src.util import get_available_device

from .lenet import LeNet5

WEIGHTS_URL = "https://www.dropbox.com/scl/fi/uoamhcvgxspbil3dek4ip/lenet_epoch-12_test_acc-0.991.pth?rlkey=20eztdion4nibm9fga3qwijbe&dl=1"


def download_if_needed(local, url):
    """Downloads a file from a URL if it does not exist locally."""
    if not os.path.exists(local):
        print(f"Downloading {url} to {local}...")
        urllib.request.urlretrieve(url, local)


def mnist_classifier(device=None) -> LeNet5:
    if device is None:
        device = get_available_device()
    net = LeNet5().eval()
    weights_file = "mnist_classifier/lenet_epoch=12_test_acc=0.991.pth"
    download_if_needed(local=weights_file, url=WEIGHTS_URL)
    net.load_state_dict(torch.load(weights_file, map_location=device))
    net = net.to(device)

    # Set requires_grad to False for all parameters
    for param in net.parameters():
        param.requires_grad = False

    return net
