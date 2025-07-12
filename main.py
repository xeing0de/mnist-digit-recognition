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

class SingleNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, NUM_CLASSES)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

model = SingleNeuron().to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

for epoch in range(0, EPOCHS):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    acc  = 100 * correct / total
    loss = epoch_loss / len(train_loader.dataset)
    print(f'Epoch {epoch:02d}: loss = {loss:.4f}, accuracy = {acc:.2f}%')

torch.save(model.state_dict(), 'single_neuron_mnist.pth')
