import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-1
DEVICE = "cpu"
NUM_CLASSES = 10

train_ds = datasets.MNIST(
    root="./data", train=True, download=True,
    transform=transforms.ToTensor()
)
test_ds = datasets.MNIST(
    root="./data", train=False, download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)
