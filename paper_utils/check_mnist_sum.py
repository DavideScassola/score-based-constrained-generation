import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.append(".")
import fnmatch

import numpy as np
from matplotlib import image
from natsort import natsorted

from mnist_classifier import get_module
from src.util import get_available_device

device = get_available_device()


def downscale(image, n=20):
    shape = np.array(image.shape) // n
    new_data = np.zeros(shape)

    for j in range(shape[0]):
        for k in range(shape[1]):
            new_data[j, k] = np.mean(image[j * n : (j + 1) * n, k * n : (k + 1) * n])

    return new_data


def image_folder_to_test_tensor(folder):
    """Converts a folder of MNIST images to a tensor for testing."""
    images = []
    files = natsorted(fnmatch.filter(names=os.listdir(folder), pat="*.png"))
    for image_path in files:
        p = Path(folder) / Path(image_path)
        image = Image.open(p).convert("L")
        image = downscale(ToTensor()(image).cpu().numpy()[0])
        images.append(image)
    return np.stack(images)


def store_images(samples: torch.Tensor, *, folder: str) -> None:
    for i, sample in enumerate(samples):
        image.imsave(
            (f"{folder}/sample_{i}.png"),
            sample.cpu().numpy(),
            cmap="gray",
        )


def classify_images_in_folder(classifier, folder):
    """Classifies an MNIST image using the given classifier."""
    x = image_folder_to_test_tensor(folder)
    x = torch.from_numpy(x).to(device).unsqueeze(1).float()

    store_images(x.squeeze(), folder="prova")

    return classifier(x).argmax(dim=-1)


def calculate_fraction_of_pairs_summing_to_10(classes):
    """Calculates the percentage of pairs of classes that sum to 10."""
    pairs = [(classes[i], classes[i + 1]) for i in range(0, len(classes), 2)]
    percentage = sum(1 for pair in pairs if sum(pair) == 10) / len(pairs)
    return percentage


paths = sys.argv[1:]

print(paths)

if not isinstance(paths, list):
    paths = [paths]


# Load the classifier
classifier = get_module.mnist_classifier(device)

for p in paths:
    folder = p + "/samples"
    if os.path.isdir(folder):
        # Classify the images in the folder
        classes = classify_images_in_folder(classifier, folder)

        # Calculate the fraction of pairs summing to 10
        fraction = calculate_fraction_of_pairs_summing_to_10(classes)

        # Store the result in a JSON file
        with open(p + "/sum10_fraction.json", "w") as f:
            json.dump({"percentage": fraction}, f)

        with open(p + "/classes.txt", "w") as f:
            for i, num in enumerate(classes.tolist()):
                print(f"{i}: {num}", file=f)
