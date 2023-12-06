import random

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    ),
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)


if __name__ == "__main__":
    # Visualization
    labels_map = {
        0: "T-shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_inx = random.randint(0, len(training_data) - 1)
        img, label = training_data[sample_inx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[torch.argmax(label).item()])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
